#!/bin/bash
python main.py --dataset "boolq" --system "unifiedqa" --strategy "qatest"
sleep 600
python main.py --dataset "boolq" --system "unifiedqa" --strategy "nocov"