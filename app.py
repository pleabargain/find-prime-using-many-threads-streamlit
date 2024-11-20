import streamlit as st
import numpy as np
import random
import threading
import time
from typing import Set
from datetime import datetime
import os
import logging
from time import perf_counter
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def clear_console():
    """Clear console based on operating system"""
    if os.name == 'nt':  # for Windows
        os.system('cls')
    else:  # for Linux/Mac
        os.system('clear')

class RunHistory:
    def __init__(self, max_entries=5):
        self.max_entries = max_entries
        self.history = []
    
    def add_run(self, num_agents, range_limit, elapsed_time, num_primes):
        """
        Add a new run to the history
        
        Parameters:
        - num_agents: Number of agents used
        - range_limit: Maximum number searched
        - elapsed_time: Time taken for the search
        - num_primes: Number of primes found
        """
        run_data = {
            'timestamp': datetime.now(),
            'num_agents': num_agents,
            'range_limit': range_limit,
            'elapsed_time': elapsed_time,
            'num_primes': num_primes
        }
        self.history.append(run_data)
        if len(self.history) > self.max_entries:
            self.history.pop(0)
    
    def get_dataframe(self):
        if not self.history:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.history)
        df['elapsed_time'] = df['elapsed_time'].round(2)
        # Ensure column order and names
        return df[['num_agents', 'range_limit', 'num_primes', 'elapsed_time']]

class AoTPrimeFinder:
    def __init__(self, range_limit: int, num_agents: int = 10):
        self.range_limit = range_limit
        self.num_agents = num_agents
        self.swarm = self._initialize_swarm()
        self.primes: Set[int] = set()
        self.lock = threading.Lock()
        self.messages = []
        self.is_running = True
        self._initialize_small_primes()

    def _is_prime_miller_rabin(self, n, k=5):
        if n <= 1:
            return False
        if n <= 3:
            return True
        if n % 2 == 0:
            return False
        
        def miller_test(d, n):
            a = random.randint(2, n - 2)
            x = pow(a, d, n)
            if x == 1 or x == n - 1:
                return True
            while d != n - 1:
                x = (x * x) % n
                d *= 2
                if x == 1:
                    return False
                if x == n - 1:
                    return True
            return False
        
        d = n - 1
        while d % 2 == 0:
            d //= 2
        
        for _ in range(k):
            if not miller_test(d, n):
                return False
        return True

    def _initialize_small_primes(self):
        for n in range(2, min(self.swarm)):
            if self._is_prime_miller_rabin(n):
                self.primes.add(n)
                self.messages.append(f"Initialized with prime: {n}")

    def stop_search(self):
        self.is_running = False

    def _initialize_swarm(self):
        return [random.randint(2, self.range_limit) for _ in range(self.num_agents)]

    def _evaluate_agent(self, agent_id: int):
        try:
            current_number = self.swarm[agent_id]
            while current_number <= self.range_limit and self.is_running:
                if self._is_prime_miller_rabin(current_number):
                    with self.lock:
                        if current_number not in self.primes:
                            self.primes.add(current_number)
                            message = f"Agent {agent_id} found prime: {current_number}"
                            logging.info(message)
                            self.messages.append(message)
                current_number += random.randint(1, 10)
                self.swarm[agent_id] = current_number
                time.sleep(0.01)
        except Exception as e:
            logging.error(f"Error in agent {agent_id}: {str(e)}")

    def find_primes(self):
        threads = []
        for agent_id in range(self.num_agents):
            thread = threading.Thread(target=self._evaluate_agent, args=(agent_id,))
            thread.daemon = True
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

def save_results(primes, num_agents, range_limit):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{num_agents}agents_{range_limit}maxprime_{timestamp}.txt"
    
    with open(filename, 'w') as f:
        f.write(f"Prime Number Search Results\n")
        f.write(f"Number of Agents: {num_agents}\n")
        f.write(f"Maximum Range: {range_limit}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Found {len(primes)} primes:\n")
        f.write(", ".join(map(str, sorted(primes))))
    
    return filename

def main():
    st.title("Agent-oriented Prime Number Finder")
    
    # Initialize run history in session state if it doesn't exist
    if 'run_history' not in st.session_state:
        st.session_state.run_history = RunHistory()

    # Clear console before each run
    clear_console()
    
    # Show run history at the top
    if st.session_state.run_history.history:
        st.subheader("Previous Runs")
        df = st.session_state.run_history.get_dataframe()
        st.dataframe(
            df,
            column_config={
                "num_agents": "Number of Agents",
                "range_limit": "Maximum Number",
                "num_primes": "Primes Found",
                "elapsed_time": "Elapsed Time (s)"
            },
            hide_index=True
        )

    st.write("""
    This application uses multiple agents to find prime numbers within a specified range.
    Each agent starts at a random position and explores the number space independently.
    Miller-Rabin primality test is used for efficient prime checking. What I find fascinating that
    using multiple agents to search for primes is slower than a single threaded approach! I added number of primes found to the UI.
    """)

    col1, col2 = st.columns(2)
    with col1:
        num_agents = st.slider("Number of Agents", min_value=1, max_value=100, value=10)
    with col2:
        range_limit = st.number_input("Maximum Number to Search", min_value=10, max_value=1000000, value=1000)

    save_output = st.checkbox("Save results to file", value=False)
    
    if st.button("Find Primes"):
        start_time = perf_counter()
        timer_placeholder = st.empty()
        progress_bar = st.progress(0)
        status_container = st.empty()
        message_container = st.empty()
        
        prime_finder = AoTPrimeFinder(range_limit, num_agents)
        
        try:
            with st.spinner('Finding prime numbers...'):
                search_thread = threading.Thread(target=prime_finder.find_primes)
                search_thread.start()
                
                while search_thread.is_alive():
                    elapsed_time = perf_counter() - start_time
                    timer_placeholder.text(f"Elapsed Time: {elapsed_time:.2f} seconds")
                    time.sleep(0.1)
                
                search_thread.join()
                final_time = perf_counter() - start_time
                found_primes = sorted(prime_finder.primes)

                # Add run to history with correct number of arguments
                st.session_state.run_history.add_run(
                    num_agents,
                    range_limit,
                    final_time,
                    len(found_primes)  # Number of primes found
                )

            st.success(f"Prime finding complete in {final_time:.2f} seconds!")
            found_primes = sorted(prime_finder.primes)
            st.write(f"Found {len(found_primes)} prime numbers:")
            
            if prime_finder.messages:
                with st.expander("Show agent activity log"):
                    for msg in prime_finder.messages:
                        st.text(msg)

            if save_output:
                filename = save_results(found_primes, num_agents, range_limit)
                st.success(f"Results saved to {filename}")

            col1, col2 = st.columns(2)
            with col1:
                st.write("First 10 primes:", found_primes[:10])
            with col2:
                st.write("Last 10 primes:", found_primes[-10:])

            with st.expander("Show all prime numbers"):
                st.write(found_primes)

            if found_primes:
                st.subheader("Distribution of Found Primes")
                hist_data = np.histogram(found_primes, bins=min(20, len(found_primes)))
                st.bar_chart(hist_data[0])

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            logging.error(f"Application error: {str(e)}")

if __name__ == "__main__":
    main()