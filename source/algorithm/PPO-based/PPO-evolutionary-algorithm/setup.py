from pathlib import Path

from setuptools import find_packages, setup


PACKAGE_ROOT = Path(__file__).resolve().parent
README_PATH = PACKAGE_ROOT / "README.md"


setup(
    name="ppo-evolutionary-autonomous-driving",
    version="0.1.0",
    description="Synchronous PPO-evolutionary highway driving baseline",
    long_description=README_PATH.read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    package_dir={"": "."},
    packages=find_packages(where=".", include=["PPOEvolutionary", "PPOEvolutionary.*"]),
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
            "ppo-evolutionary-train=PPOEvolutionary.scripts.train:main",
            "ppo-evolutionary-evaluate=PPOEvolutionary.scripts.evaluate:main",
        ],
    },
    zip_safe=False,
)
