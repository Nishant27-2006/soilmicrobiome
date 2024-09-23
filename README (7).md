
# Autonomous Soil Microbiome Management using Random Forest and Deep Reinforcement Learning

## Overview

This project presents a system for managing soil microbiomes using weather data, Random Forest models, and Deep Reinforcement Learning (DRL). By predicting key soil-related outcomes (such as temperature, microbial diversity, or moisture) based on environmental data, this system can optimize interventions to enhance soil health in real-time.

### Key Features:
- **Random Forest Model**: Used to predict soil health indicators based on weather data (temperature, dew point, precipitation, etc.).
- **Deep Reinforcement Learning (DRL)**: Manages soil health by determining optimal interventions, such as irrigation adjustments and nutrient applications.
- **Real-time feedback loop**: Continuously evaluates and improves predictions and interventions based on real-time data.

## Dataset

This project uses NOAA GSOD data for environmental variables such as temperature, precipitation, and dew point to predict soil health and guide interventions.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/soil-microbiome-management.git
   ```

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. (Optional) Set up a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # For Linux/macOS
   .\venv\Scripts\activate  # For Windows
   ```

## Usage

1. Place your dataset (CSV format) in the project directory.
2. Run the main script to train the Random Forest model and visualize results:
   ```bash
   python main.py
   ```

3. View the model performance metrics and visualizations.

## Visualizations
- **True vs Predicted Temperature**: A scatter plot comparing true and predicted temperature values.
- **Feature Importance**: A bar chart showing the importance of each feature in the prediction model.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
