import React, { useState } from 'react';
import './App.css';
import ForecastForm from './components/ForecastForm';
import ForecastResults from './components/ForecastResults';

function App() {
  const [loading, setLoading] = useState(false);
  const [forecastData, setForecastData] = useState(null);
  const [darkMode, setDarkMode] = useState(() => {
    return localStorage.getItem('theme') === 'dark';
  });

  const toggleDarkMode = () => {
    const newMode = !darkMode;
    setDarkMode(newMode);
    localStorage.setItem('theme', newMode ? 'dark' : 'light');
  };

  const generateForecast = async (formData) => {
    setLoading(true);
    setForecastData(null);
    try {
      const res = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData)
      });

      if (!res.ok) {
        const errorData = await res.json();
        throw new Error(errorData.error || 'Failed to generate forecast');
      }

      const data = await res.json();
      setForecastData(data);
    } catch (err) {
      console.error('Error during forecast request:', err);
      alert('Error: ' + (err.message || 'Failed to generate forecast'));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className={`app ${darkMode ? 'dark' : ''}`}>
      <button className="theme-toggle" onClick={toggleDarkMode}>
        {darkMode ? '☀️ Light Mode' : '🌙 Dark Mode'}
      </button>
      <div className="container">
        <h1>Climate Forecast</h1>
        <ForecastForm onSubmit={generateForecast} loading={loading} />
        {forecastData && <ForecastResults data={forecastData} />}
      </div>
    </div>
  );
}

export default App;
