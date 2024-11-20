# Agent-oriented Prime Number Finder


This experiment was created to test the performance of a multi-agent system for finding prime numbers after watching this video: https://www.youtube.com/watch?v=9NcYZD6mLag

I found the results surprising. I expected the multi-agent system to perform better than the single-threaded approach, but the results show that the single-threaded approach is faster. I expected the multi-agent system to find more primes, but the results show that the single-threaded approach finds primes faster. Why is this?

here is the code for the experiment:
https://github.com/pleabargain/find-prime-using-many-threads-streamlit




A Streamlit web application that implements a multi-agent approach to finding prime numbers within a specified range.

https://pleabargain-find-prime-using-many-threads-streamlit-app-w29spr.streamlit.app/

## Description

This application demonstrates a parallel computing approach to prime number discovery using multiple autonomous agents. Each agent independently explores different parts of the number space to find prime numbers efficiently.

### Features

- Multi-threaded agent-based computation
- Interactive web interface built with Streamlit
- Real-time progress tracking
- Visual distribution analysis of found primes
- Configurable number of agents and search range

## Technical Details

The application consists of two main components:

1. **AoTPrimeFinder Class**
   - Manages a swarm of agents searching for prime numbers
   - Implements thread-safe prime number discovery
   - Uses a lock mechanism to prevent race conditions
   - Each agent explores the number space with random increments

2. **Streamlit Interface**
   - Allows users to configure:
     - Number of agents (1-100)
     - Maximum search range (10-1,000,000)
   - Displays real-time updates from each agent
   - Shows final results with:
     - List of discovered primes
     - Histogram of prime number distribution

## Usage

1. Adjust the number of agents using the slider
2. Set the maximum number to search
3. Click "Find Primes" to start the computation
4. Watch real-time updates as agents discover prime numbers
5. View the final results and distribution visualization

## Requirements

- Python 3.x
- Streamlit
- NumPy
- Threading (standard library)

## Installation


python -m streamlit run app.py
