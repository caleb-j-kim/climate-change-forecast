import React, { useState, useEffect } from 'react';

function ForecastForm({ onSubmit, loading }) {
    const [locationType, setLocationType] = useState('country');
    const [locations, setLocations] = useState([]);
    const [selectedLocation, setSelectedLocation] = useState('');
    const [year, setYear] = useState(new Date().getFullYear());
    const [month, setMonth] = useState(1);
    const [loadingLocations, setLoadingLocations] = useState(false);

    // Fetch locations when location type changes
    useEffect(() => {
        const fetchLocations = async () => {
            setLoadingLocations(true);
            try {
                const res = await fetch(`http://localhost:5000/locations?type=${locationType}`);
                const data = await res.json();
                setLocations(data);

                // Reset selected location
                setSelectedLocation('');
            } catch (err) {
                console.error('Error fetching locations:', err);
                alert('Failed to load locations. Please try again.');
            } finally {
                setLoadingLocations(false);
            }
        };

        fetchLocations();
    }, [locationType]);

    const handleSubmit = (e) => {
        e.preventDefault();

        // Process location data
        let location;

        if (locationType === 'country') {
            location = selectedLocation;
        } else {
            try {
                location = JSON.parse(selectedLocation);
            } catch (err) {
                console.error(`Error parsing ${locationType} JSON:`, err);
                alert(`Invalid ${locationType} data format`);
                return;
            }
        }

        onSubmit({
            dataset: locationType,
            location,
            year,
            month
        });
    };

    const handleLocationChange = (e) => {
        const value = e.target.value;
        setSelectedLocation(value);
        // We'll use the option text directly from the select when needed
    };

    return (
        <form onSubmit={handleSubmit}>
            <div className="form-group">
                <label htmlFor="type">Location Type:</label>
                <select
                    id="type"
                    value={locationType}
                    onChange={(e) => setLocationType(e.target.value)}
                    required
                >
                    <option value="country">Country</option>
                    <option value="state">State</option>
                    <option value="city">City</option>
                </select>
            </div>

            <div className="form-group">
                <label htmlFor="location">Location:</label>
                <select
                    id="location"
                    value={selectedLocation}
                    onChange={handleLocationChange}
                    required
                    disabled={loadingLocations}
                >
                    <option value="">Select {locationType}</option>
                    {locations.map((loc, index) => {
                        if (locationType === 'country') {
                            return (
                                <option key={index} value={loc}>
                                    {loc}
                                </option>
                            );
                        } else if (locationType === 'state') {
                            return (
                                <option key={index} value={JSON.stringify(loc)}>
                                    {loc.state}, {loc.country}
                                </option>
                            );
                        } else if (locationType === 'city') {
                            return (
                                <option key={index} value={JSON.stringify(loc)}>
                                    {loc.city}, {loc.country}
                                </option>
                            );
                        }
                        return null;
                    })}
                </select>
                {loadingLocations && <div className="spinner-small"></div>}
            </div>

            <div className="form-group">
                <label htmlFor="year">Year:</label>
                <input
                    type="number"
                    id="year"
                    value={year}
                    onChange={(e) => setYear(parseInt(e.target.value, 10))}
                    required
                />
            </div>

            <div className="form-group">
                <label htmlFor="month">Month:</label>
                <select
                    id="month"
                    value={month}
                    onChange={(e) => setMonth(parseInt(e.target.value, 10))}
                    required
                >
                    <option value="1">Jan</option>
                    <option value="2">Feb</option>
                    <option value="3">Mar</option>
                    <option value="4">Apr</option>
                    <option value="5">May</option>
                    <option value="6">Jun</option>
                    <option value="7">Jul</option>
                    <option value="8">Aug</option>
                    <option value="9">Sep</option>
                    <option value="10">Oct</option>
                    <option value="11">Nov</option>
                    <option value="12">Dec</option>
                </select>
            </div>

            <button
                type="submit"
                className="submit-button"
                disabled={loading || !selectedLocation}
            >
                {loading ? 'Generating...' : 'Generate Forecast'}
            </button>
            {loading && <div className="spinner"></div>}
        </form>
    );
}

export default ForecastForm;