from pathlib import Path

from setuptools import find_packages, setup


PACKAGE_ROOT = Path(__file__).resolve().parent
README_PATH = PACKAGE_ROOT / "AlphaZero" / "README.md"


setup(
    name="alphazero-autonomous-driving",
    version="0.1.0",
    description="AlphaZero-style autonomous driving package on top of highway-env",
    long_description=README_PATH.read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    package_dir={"": "."},
    packages=find_packages(where=".", include=["AlphaZero", "AlphaZero.*"]),
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
            "alphazero-self-play=AlphaZero.scripts.self_play:main",
            "alphazero-self-play-racetrack=AlphaZero.scripts.self_play_parallel_racetrack:main",
            "alphazero-train-from-self-play=AlphaZero.scripts.train_from_self_play:main",
            "alphazero-progressive-self-play=AlphaZero.scripts.progressive_self_play:main",
            "alphazero-progressive-train=AlphaZero.scripts.progressive_train:main",
            "alphazero-infer=AlphaZero.scripts.infer:main",
            "alphazero-evaluate=AlphaZero.scripts.evaluate:main",
        ],
    },
    zip_safe=False,
)
