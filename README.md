# SB-ideaForce-SN

## Project Description

This project analyzes data from My Starbucks Idea, focusing on user interactions, suggestions, and comments. It constructs complex networks to study user-idea relationships, user-user interactions, and idea-idea similarities.

## Features

- **Data Ingestion & Cleaning**: Parse timestamps, normalize author names, and map author roles.
- **Network Construction**: Build user-idea bipartite graphs, user-user interaction graphs, and idea-idea similarity graphs.
- **Community Detection**: Identify user communities and analyze their characteristics.
- **Inter-Role Influence Analysis**: Compute centrality measures and analyze influence patterns.
- **Idea Success & Lifetime Modeling**: Predict idea success and lifetime using various features.

## Setup

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/Abhi-rangi/SB-ideaForce-SN.git
   cd SB-ideaForce-SN
   ```

2. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Database**:
   - Update `config.properties` with your database password.

## Usage

1. **Run the Script**:

   ```bash
   python script.py
   ```

2. **Network Construction**:
   ```bash
   python network_construction.py
   ```

## Contributing

1. Fork the repository.
2. Create a new branch for your feature.
3. Commit your changes.
4. Push to the branch.
5. Create a Pull Request.

