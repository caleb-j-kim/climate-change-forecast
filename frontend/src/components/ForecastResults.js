import React from 'react';
import { Line } from 'react-chartjs-2';
import {
    Chart as ChartJS,
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    Title,
    Tooltip,
    Legend,
    Filler
} from 'chart.js';

// Register Chart.js components
ChartJS.register(
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    Title,
    Tooltip,
    Legend,
    Filler
);

function ForecastResults({ data }) {
    // Extract location information based on data format
    let locationDisplay = '';
    if (typeof data.location === 'string') {
        locationDisplay = data.location;
    } else if (typeof data.location === 'object') {
        if (data.location.city) {
            locationDisplay = `${data.location.city}, ${data.location.country}`;
        } else if (data.location.state) {
            locationDisplay = `${data.location.state}, ${data.location.country}`;
        } else if (data.location.country) {
            locationDisplay = data.location.country;
        }
    }

    const chartData = {
        labels: ['Linear Regression', 'Random Forest', 'Ensemble'],
        datasets: [
            {
                label: `Temperature (°C) in ${locationDisplay} (${data.month}/${data.year})`,
                data: [
                    data.lr_prediction,
                    data.rf_prediction,
                    data.ensemble_prediction
                ],
                borderColor: '#0077ff',
                backgroundColor: 'rgba(0,119,255,0.1)',
                tension: 0.4,
                fill: true,
                pointBackgroundColor: '#0077ff'
            }
        ]
    };

    const chartOptions = {
        responsive: true,
        plugins: {
            title: {
                display: true,
                text: 'Forecast Model Outputs',
                font: { size: 16 }
            },
            legend: { display: false }
        },
        scales: {
            y: {
                beginAtZero: true,
                title: { display: true, text: 'Temperature (°C)' }
            }
        }
    };

    return (
        <div className="forecast-results">
            <div className="summary-section">
                <h2>Forecast Summary</h2>
                <p><strong>Location:</strong> {locationDisplay}</p>
                <p><strong>Date:</strong> {data.month}/{data.year}</p>
                <p>Linear Regression: {data.lr_prediction.toFixed(2)} °C</p>
                <p>Random Forest: {data.rf_prediction.toFixed(2)} °C</p>
                <p>
                    <strong>Ensemble Avg:</strong>
                    <span className="highlight-value">
                        {data.ensemble_prediction.toFixed(2)} °C
                    </span>
                </p>
            </div>

            {data.summary && (
                <div className="ai-summary">
                    <h3>AI Summary</h3>
                    <p>{data.summary}</p>
                </div>
            )}

            <div className="chart-container">
                <Line data={chartData} options={chartOptions} />
            </div>
        </div>
    );
}

export default ForecastResults;