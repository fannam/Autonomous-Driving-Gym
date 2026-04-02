from pathlib import Path

from setuptools import find_packages, setup


PACKAGE_ROOT = Path(__file__).resolve().parent
README_PATH = PACKAGE_ROOT / "README.md"


setup(
    name="alphazero-meta-adversarial-autonomous-driving",
    version="0.1.0",
    description="Adversarial multi-agent AlphaZero with DiscreteMetaAction on highway-env",
    long_description=README_PATH.read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    package_dir={"": "."},
    packages=find_packages(where=".", include=["AlphaZeroMetaAdversarial", "AlphaZeroMetaAdversarial.*"]),
    include_package_data=True,
    python_requires=">=3.12",
    install_requires=[
        "gymnasium>=1.2.3",
        "highway-env",
        "numpy>=2.4.2",
        "PyYAML>=6.0.2",
        "torch>=2.10.0",
    ],
    entry_points={
        "console_scripts": [
            "alphazero-meta-adversarial-self-play=AlphaZeroMetaAdversarial.scripts.self_play:main",
            "alphazero-meta-adversarial-self-play-kaggle-dual-gpu=AlphaZeroMetaAdversarial.scripts.self_play_kaggle_dual_gpu:main",
            "alphazero-meta-adversarial-train=AlphaZeroMetaAdversarial.scripts.train:main",
            "alphazero-meta-adversarial-evaluate=AlphaZeroMetaAdversarial.scripts.evaluate:main",
        ],
    },
    zip_safe=False,
)
