import React, { useState } from 'react';
import './App.css';
import ForecastForm from './components/ForecastForm';
import ForecastResults from './components/ForecastResults';
import Iridescence from './reactbits/Iridescence';



function App() {
  const [loading, setLoading] = useState(false);
  const [forecastData, setForecastData] = useState(null);


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
    <div className={`app`}>
      <div style={{ width: '100%', height: '50px', depth: '50px', position: 'relative' }}>
        <Iridescence
          color={[1, 1, 1]}
          mouseReact={false}
          amplitude={0.1}
          speed={1.0}
        />
      </div>
      <div className="container">
        <h1>Climate Forecast</h1>
        <ForecastForm onSubmit={generateForecast} loading={loading} />
        {forecastData && <ForecastResults data={forecastData} />}
      </div>
    </div>
  );
}

export default App;
