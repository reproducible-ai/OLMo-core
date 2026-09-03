"""One-GPU capture of the upstream OLMo 3 scaling-ladder 3B rung.

The model and optimizer come from OLMo-core's published ladder. This adapter makes
the topology and bounded-run changes used on one RTX PRO 6000 Blackwell Server
Edition and prepares a deterministic subset of Ai2's public Dolma 3 training mix.
"""

import argparse
import io
import json
import tarfile
from pathlib import Path

import numpy as np


DATASET_REVISION = "689a3ea2d8217e64d73a5058913fa43ad15e81aa"
DATASET_FILE = "data/dolma1_7-wiki-en/00000.jsonl.zst"
TOKENIZER_REVISION = "5292e5d6c0f40b67cc765fe41bec991cf4345b5c"
SEQUENCE_LENGTH = 8192


def prepare_data(output: Path, manifest: Path, num_tokens: int) -> None:
    import requests
    import zstandard
    from transformers import AutoTokenizer

    url = (
        "https://huggingface.co/datasets/allenai/dolma3_mix-6T/resolve/"
        f"{DATASET_REVISION}/{DATASET_FILE}"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "allenai/dolma2-tokenizer", revision=TOKENIZER_REVISION
    )
    token_ids: list[int] = []
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    with zstandard.ZstdDecompressor().stream_reader(response.raw) as reader:
        text_stream = io.TextIOWrapper(reader, encoding="utf-8")
        for line in text_stream:
            document = json.loads(line)
            token_ids.extend(tokenizer.encode(document["text"], add_special_tokens=False))
            token_ids.append(100257)
            if len(token_ids) >= num_tokens:
                break

    if len(token_ids) < num_tokens:
        raise RuntimeError(f"public shard supplied only {len(token_ids)} tokens")
    output.parent.mkdir(parents=True, exist_ok=True)
    array = np.memmap(output, mode="w+", dtype=np.uint32, shape=(num_tokens,))
    array[:] = token_ids[:num_tokens]
    array.flush()
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(
            {
                "dataset": "allenai/dolma3_mix-6T",
                "datasetRevision": DATASET_REVISION,
                "sourceFile": DATASET_FILE,
                "tokenizer": "allenai/dolma2-tokenizer",
                "tokenizerRevision": TOKENIZER_REVISION,
                "numTokens": num_tokens,
                "dtype": "uint32",
            },
            indent=2,
        )
        + "\n"
    )


def train(data: Path, output_dir: Path) -> None:
    from olmo_core.data import TokenizerConfig
    from olmo_core.data.composable import (
        ComposableDataLoaderConfig,
        ConcatAndChunkInstanceSourceConfig,
        InstanceFilterConfig,
    )
    from olmo_core.model_ladder import (
        DeviceMeshSpec,
        ModelLadder,
        Olmo3ModelConfigurator,
        WSDSChinchillaRunConfigurator,
    )
    from olmo_core.train import Duration

    class OneGpuOlmo3Configurator(Olmo3ModelConfigurator):
        def configure_minimal_device_mesh_spec(self, **kwargs) -> DeviceMeshSpec:
            return DeviceMeshSpec(world_size=1, dp_world_size=None)

    class OneStepRunConfigurator(WSDSChinchillaRunConfigurator):
        def configure_target_batch_size(self, num_params: int) -> int:
            del num_params
            return SEQUENCE_LENGTH

        def configure_duration(self, num_params: int, batch_size: int) -> Duration:
            del num_params, batch_size
            return Duration.steps(1)

        def configure_checkpoint_intervals(self, num_params: int, batch_size: int):
            del num_params, batch_size
            return [(Duration.steps(1), "bounded capture")]

    class CaptureLadder(ModelLadder):
        def _configure_trainer(self, size_spec: str, for_benchmarking: bool = False):
            trainer = super()._configure_trainer(size_spec, for_benchmarking)
            trainer.callbacks["lm_evaluator"].enabled = False
            trainer.callbacks["downstream_evaluator"].enabled = False
            trainer.callbacks["wandb"].enabled = False
            trainer.callbacks["checkpointer"].pre_train_checkpoint = False
            trainer.callbacks["checkpointer"].save_async = False
            return trainer

    tokenizer = TokenizerConfig.dolma2()
    ladder = CaptureLadder(
        name="olmo3-ladder-row027",
        project="row027-olmo-core",
        dir=str(output_dir),
        sizes=["3B"],
        max_devices=1,
        device_type="NVIDIA B200",
        model_configurator=OneGpuOlmo3Configurator(rank_microbatch_size=SEQUENCE_LENGTH),
        run_configurator=OneStepRunConfigurator(chinchilla_multiple=4.0),
        sequence_length=SEQUENCE_LENGTH,
        tokenizer=tokenizer,
        instance_sources=[
            ConcatAndChunkInstanceSourceConfig.from_npy(
                str(data), tokenizer=tokenizer, sequence_length=SEQUENCE_LENGTH
            )
        ],
        data_loader=ComposableDataLoaderConfig(
            num_workers=0, instance_filter_config=InstanceFilterConfig()
        ),
    )
    ladder.run("3B")


def package_checkpoint(checkpoint: Path, output: Path) -> None:
    paths = sorted(checkpoint.rglob("*"), key=lambda path: path.as_posix())
    if not paths:
        raise RuntimeError(f"checkpoint directory is empty: {checkpoint}")
    output.parent.mkdir(parents=True, exist_ok=True)
    expected_members: list[str] = []
    with tarfile.open(output, mode="w", format=tarfile.PAX_FORMAT) as archive:
        for path in paths:
            archive_name = path.relative_to(checkpoint.parent).as_posix()
            info = archive.gettarinfo(path, arcname=archive_name)
            info.mtime = 0
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            expected_members.append(archive_name)
            if path.is_file():
                with path.open("rb") as source:
                    archive.addfile(info, source)
            else:
                archive.addfile(info)
    with tarfile.open(output, mode="r") as archive:
        actual_members = archive.getnames()
    if actual_members != expected_members:
        raise RuntimeError("checkpoint archive member verification failed")
    print(f"packaged {len(actual_members)} checkpoint entries into {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    prep = subparsers.add_parser("prepare-data")
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--manifest", type=Path, required=True)
    prep.add_argument("--num-tokens", type=int, default=32768)
    run = subparsers.add_parser("train")
    run.add_argument("--data", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    package = subparsers.add_parser("package-checkpoint")
    package.add_argument("--checkpoint", type=Path, required=True)
    package.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare-data":
        prepare_data(args.output, args.manifest, args.num_tokens)
    elif args.command == "train":
        train(args.data, args.output_dir)
    else:
        package_checkpoint(args.checkpoint, args.output)


if __name__ == "__main__":
    main()
