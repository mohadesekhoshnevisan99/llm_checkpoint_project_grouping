from .crossjob import CrossJobPeerStrategy, DonorRegistry
from .object_store import ObjectStoreCheckpointStrategy
from .tiered import TieredCheckpointStrategy


def create_checkpoint_strategy(name, **kwargs):
    if name == "crossjob_peer":
        return CrossJobPeerStrategy(**kwargs)
    if name == "object_store":
        kwargs.pop("workers", None)
        return ObjectStoreCheckpointStrategy(**kwargs)
    if name in {
        "cpu_tiered",
        "ssd_tiered",
        "local_tiered",
        "paired_tiered",
    }:
        paired = name in {
            "cpu_tiered",
            "ssd_tiered",
            "paired_tiered",
        }
        return TieredCheckpointStrategy(
            paired=paired,
            persist_to_ssd=name != "cpu_tiered",
            strategy_name=(
                "ssd_tiered" if name == "local_tiered" else name
            ),
            **kwargs,
        )
    raise ValueError(f"Unsupported checkpoint strategy: {name}")


__all__ = [
    "CrossJobPeerStrategy",
    "DonorRegistry",
    "ObjectStoreCheckpointStrategy",
    "TieredCheckpointStrategy",
    "create_checkpoint_strategy",
]
