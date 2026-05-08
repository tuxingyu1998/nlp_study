"""
语序信息对比实验：「你」在第几位 → 第几类（1~5）

模型 A：Embedding → RNN → MaxPool → Linear（保留语序线索）
模型 B：Embedding → MaxPool → Linear（词袋式，与顺序无关）

依赖：pip install torch>=2.0
"""

from __future__ import annotations

import argparse
import random
import sys
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# 数据：五字句 + 「你」的位置标签。词袋视角下，只要 multiset（字及其频次）相同，
# 顺序不同的句子会得到相同的 MaxPool 表征，因此 BoW 路径难以学到「第几位」。
# ---------------------------------------------------------------------------


def _zh_pool() -> List[str]:
    """不含「你」的常用字池，用于填充其余 4 个位置。"""
    # 字池中剔除「你」，避免除目标位置外又随机抽到「你」，保证句中只有一个「你」
    s = (
        "的一是在不了有和人这中大为上个国我以要他时来用们生到作地于出就分对成会"
        "可主发年动同工也能下过子说产种面而方后多定行学法所民得经十三之进着等"
        "部度家电力里如水化高自二理起小物现实加量都两体制机当使点从业本去把性"
        "好应开它合还因由其些然前外天政四日那社义事平形相全表间样与关各重新线"
        "内数正心反你明看原又么利比或但质气第向道命此变条只没结解问意建月公无"
        "系军很情者最立代想已通并提直题党程展五果料象员革位入常文总次品式活设"
        "及管特件长求老头基资边流路级少图山统接知较将组见计别她手角期根论运农"
        "指几九区强放决西被干做必战先回则任取据处队南给色光门即保治北造百规热"
        "领七海口东导器压志世金增争济阶油思术极交受联什认六共权收证改清己美再"
        "采转更单风切打白教速花带安场身车例真务具万每目至达走积示议声报斗完类"
        "八离华名确才科张信马节话米整空元况今集温传土许步群广石记需段研界拉林"
        "律叫且究观越织装影算低持音众书布复容般须始旅哪卫派央叶操伟退讨奋昨仿"
        "赞雾晋狼寸晴厦彦桓玄姬舜珑琛涅皋塾殆攸泓胛箴臾攫辘忻邕旖胛迸篝隍玷淬"
    )
    return [c for c in s if c != "你"]


class PositionSentenceDataset(Dataset):
    """
    随机生成含「你」的 5 字句；标签为「你」的位置（1~5，转为 0~4）。
    其余位置从字池中随机抽取，保证句中恰有一个「你」。
    """

    def __init__(self, num_samples: int, seed: int, pool: List[str] | None = None):
        super().__init__()
        self.pool = pool or _zh_pool()
        if len(self.pool) < 1:
            raise ValueError("character pool must be non-empty")
        rng = random.Random(seed)
        self.sentences: List[str] = []
        self.labels: List[int] = []
        for _ in range(num_samples):
            # 类别 1~5：「你」在第 1~5 字；存标签时用 0~4 对应 CrossEntropyLoss
            pos = rng.randint(1, 5)
            chars = [rng.choice(self.pool) for _ in range(5)]
            chars[pos - 1] = "你"
            self.sentences.append("".join(chars))
            self.labels.append(pos - 1)

    def __len__(self) -> int:
        return len(self.sentences)

    def __getitem__(self, idx: int) -> Tuple[str, int]:
        return self.sentences[idx], self.labels[idx]


def build_vocab(pool: List[str]) -> dict:
    # 「你」单独列出；其余字去重排序，得到稳定的小词表
    chars = ["你"] + sorted(set(pool))
    return {c: i for i, c in enumerate(chars)}


def collate_fn(
    batch: List[Tuple[str, int]], char_to_idx: dict, device: torch.device
):
    """字符串 batch → 长整型 (batch, 5)，已在目标 device 上。"""
    max_len = 5
    x = torch.zeros(len(batch), max_len, dtype=torch.long, device=device)
    y = torch.tensor([b[1] for b in batch], dtype=torch.long, device=device)
    for i, (sent, _) in enumerate(batch):
        for t, ch in enumerate(sent[:max_len]):
            x[i, t] = char_to_idx[ch]
    return x, y


class RNNMaxPoolClassifier(nn.Module):
    """Embedding → RNN → MaxPool(时间维) → Linear"""

    # RNN 按字递推，隐藏状态依赖前缀顺序；再对时间维取 max 得到句向量，仍携带位置线索

    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        embed_dim: int,
        hidden_dim: int,
        rnn_type: str = "GRU",
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        rnn_cls = nn.GRU if rnn_type.upper() == "GRU" else nn.RNN
        self.rnn = rnn_cls(
            embed_dim,
            hidden_dim,
            batch_first=True,
            num_layers=1,
        )
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 5)；e/out: (batch, 5, dim)；pooled: (batch, hidden_dim)
        e = self.embed(x)
        out, _ = self.rnn(e)
        pooled, _ = out.max(dim=1)  # 沿时间维池化，输入顺序已反映在 out 中
        return self.fc(pooled)


class BoWMaxPoolClassifier(nn.Module):
    """Embedding → MaxPool(时间维) → Linear（无 RNN，近似词袋）"""

    # 仅对「字的 embedding」做 max，无序列建模；交换字序不改变 multiset 上的 max 组合性质（典型词袋失效场景）

    def __init__(self, vocab_size: int, num_classes: int, embed_dim: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e = self.embed(x)  # (batch, 5, embed_dim)
        pooled, _ = e.max(dim=1)  # 逐维取 max，不依赖字的出现顺序
        return self.fc(pooled)


@torch.no_grad()
def accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """验证集分类准确率（collate 已把张量放到 device）。"""
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        logits = model(x)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    opt: torch.optim.Optimizer,
    loss_fn: nn.Module,
) -> float:
    """单轮训练，返回样本平均交叉熵。"""
    model.train()
    total_loss = 0.0
    n = 0
    for x, y in loader:
        opt.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        opt.step()
        total_loss += loss.item() * x.size(0)
        n += x.size(0)
    return total_loss / max(n, 1)


def main() -> None:
    # 两个模型独立训练、同一数据划分，便于公平对比验证准确率
    p = argparse.ArgumentParser(description="语序对比：RNN+MaxPool vs BoW+MaxPool")
    p.add_argument("--train-samples", type=int, default=20000)
    p.add_argument("--val-samples", type=int, default=4000)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--embed-dim", type=int, default=64)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--rnn-type", choices=("GRU", "RNN"), default="GRU")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pool = _zh_pool()
    char_to_idx = build_vocab(pool)
    vocab_size = len(char_to_idx)
    num_classes = 5

    train_ds = PositionSentenceDataset(args.train_samples, seed=args.seed, pool=pool)
    # 验证集用不同种子重新采样，避免与训练集句子重合
    val_ds = PositionSentenceDataset(
        args.val_samples, seed=args.seed + 9999, pool=pool
    )

    def _collate(batch):
        # 闭包捕获 char_to_idx 与 device，DataLoader worker 若 num_workers>0 需可 pickle（此处默认 0）
        return collate_fn(batch, char_to_idx, device)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=_collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=_collate,
    )

    models = {
        "A_RNN_MaxPool": RNNMaxPoolClassifier(
            vocab_size,
            num_classes,
            args.embed_dim,
            args.hidden_dim,
            rnn_type=args.rnn_type,
        ).to(device),
        "B_BoW_MaxPool": BoWMaxPoolClassifier(
            vocab_size, num_classes, args.embed_dim
        ).to(device),
    }

    loss_fn = nn.CrossEntropyLoss()  # 标签 0~4，五分类
    random_baseline = 100.0 / num_classes  # 均匀先验下随机猜：20%

    print("=" * 60)
    print("语序信息对比实验：「你」在 5 字句中的位置分类（5 类）")
    print(f"设备: {device} | 训练/验证样本: {args.train_samples}/{args.val_samples}")
    print(f"随机猜测基准准确率: {random_baseline:.1f}%")
    print("=" * 60)

    for name, model in models.items():
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        print(f"\n>>> {name}")
        for ep in range(1, args.epochs + 1):
            tr_loss = train_one_epoch(model, train_loader, opt, loss_fn)
            va_acc = accuracy(model, val_loader, device)
            # 首轮、末轮及约每 1/5 轮打印，避免输出过长
            if ep == 1 or ep == args.epochs or ep % max(1, args.epochs // 5) == 0:
                print(
                    f"  epoch {ep:3d}/{args.epochs}  "
                    f"train_loss={tr_loss:.4f}  val_acc={va_acc*100:.2f}%"
                )
        final_acc = accuracy(model, val_loader, device)
        print(f"  最终验证准确率: {final_acc*100:.2f}%")

    print("\n" + "=" * 60)
    print("解读：RNN 逐步编码上下文，MaxPool 后仍能区分「你」的位置；")
    print("BoW（字袋 MaxPool）丢弃顺序，理论上难以超过随机基准 (~20%)。")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
