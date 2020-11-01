#!/usr/bin/env bash


echo 'Running all samplers: ULA, MALA, and SGLD..'

echo 'Running ULA'
python run_ULA_increase_data.py
python run_ULA_increase_dimension.py

echo 'Running MALA'
python run_MALA_increase_data.py
python run_MALA_increase_dimension.py

echo 'Running SGLD'
python run_SGLD_increase_data.py
python run_SGLD_increase_dimension.py
