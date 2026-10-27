#!/usr/bin/env bash
# Downloads the three public IoT datasets used in this study into ./data
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p data && cd data

# A / D: household smart-meter readings (UCI), mirrored on GitHub
curl -fSL -o household_power_consumption.zip \
  https://raw.githubusercontent.com/jbrownlee/Datasets/master/household_power_consumption.zip
unzip -o -q household_power_consumption.zip && rm household_power_consumption.zip

# B: daily minimum temperatures, Melbourne 1981-1990
curl -fSL -o daily-min-temperatures.csv \
  https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv

# C: office-room occupancy detection (UCI), author's GitHub copy
for f in datatraining.txt datatest.txt datatest2.txt; do
  curl -fSL -o "$f" "https://raw.githubusercontent.com/LuisM78/Occupancy-detection-data/master/$f"
done

ls -la
