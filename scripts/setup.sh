#!/usr/bin/env bash
# Installs dependencies and fetches the official LaMP-QA evaluator.
set -euo pipefail
cd "$(dirname "$0")/.."

pip install -r requirements.txt

mkdir -p third_party
if [ ! -d third_party/LaMP-QA ]; then
  git clone https://github.com/LaMP-Benchmark/LaMP-QA.git third_party/LaMP-QA
fi
echo "Setup done. LaMP-QA evaluator is in third_party/LaMP-QA."
