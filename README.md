# Space Weather Dashboard 🌌

A comprehensive Streamlit dashboard for visualizing and analyzing space weather data, with special focus on potential correlations between solar activity and health patterns.

## Features

- **Multi-parameter Space Weather Visualization**: Kp index, sunspot numbers, solar flares, and more
- **Geomagnetic Activity Analysis**: Storm threshold monitoring and classification
- **Solar Activity Tracking**: Real-time solar flux and sunspot analysis
- **Health Correlation Analysis**: Statistical analysis of space weather vs. migraine episodes
- **Interactive Timeline**: Visual correlation between space weather events and health data
- **Statistical Analysis**: T-tests, correlation analysis, and significance testing

## Live Dashboard

🚀 **[View Live Dashboard](https://your-app-name.streamlit.app)**

## Data Sources

This dashboard uses daily space weather index files from NOAA, covering:
- March 21, 2025 to June 3, 2025
- Kp geomagnetic indices
- Sunspot numbers
- Solar radio flux data
- Solar flare classifications

## Local Development

```bash
# Clone the repository
git clone https://github.com/yourusername/space-weather-dashboard.git
cd space-weather-dashboard

# Install dependencies
pip install -r requirements.txt

# Run the dashboard
streamlit run space_weather_dashboard.py
```

## Health Analysis

The dashboard includes a specialized health analysis section that:
- Tracks migraine episodes on specific dates
- Compares space weather conditions between migraine and non-migraine days
- Provides statistical significance testing
- Offers research-based context and recommendations

**Note**: This analysis is for informational purposes only and should not replace medical advice.

## Technologies Used

- **Streamlit**: Interactive web dashboard framework
- **Plotly**: Advanced interactive visualizations
- **Pandas**: Data manipulation and analysis
- **SciPy**: Statistical analysis and hypothesis testing
- **Matplotlib/Seaborn**: Additional plotting capabilities

## Contributing

Feel free to submit issues, feature requests, or pull requests to improve the dashboard.

## License

MIT License - See LICENSE file for details.

---

*Built with ❤️ for space weather enthusiasts and health researchers*
