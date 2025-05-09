import React, { useState, useEffect } from 'react';
import BlurText from "../reactbits/BlurText";

function ForecastForm({ onSubmit, loading }) {
    const [locationType, setLocationType] = useState('country');
    const [locations, setLocations] = useState([]);
    const [selectedLocation, setSelectedLocation] = useState('');
    const [year, setYear] = useState(new Date().getFullYear());
    const [month, setMonth] = useState(1);
    const [loadingLocations, setLoadingLocations] = useState(false);

    useEffect(() => {
        const fetchLocations = async () => {
            setLoadingLocations(true);
            try {
                const res = await fetch(`http://localhost:5000/locations?type=${locationType}`);
                const data = await res.json();
                setLocations(data);
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

    const handleLocationTypeChange = (e) => {
        const newType = e.target.value;
        setLocationType(newType);
        setSelectedLocation('');
    };

    const handleLocationChange = (e) => {
        const value = e.target.value;
        setSelectedLocation(value);
    };

    const handleSubmit = (e) => {
        e.preventDefault();

        let location;

        if (locationType === 'country') {
            if (typeof selectedLocation === 'string' && selectedLocation.startsWith('{')) {
                try {
                    const parsed = JSON.parse(selectedLocation);
                    location = parsed.country;
                } catch {
                    location = selectedLocation;
                }
            } else {
                location = selectedLocation;
            }
        } else {
            try {
                location = typeof selectedLocation === 'string' && selectedLocation.startsWith('{')
                    ? JSON.parse(selectedLocation)
                    : selectedLocation;
            } catch (err) {
                console.error(`Error parsing ${locationType} JSON:`, err);
                alert(`Invalid ${locationType} data format`);
                return;
            }
        }

        console.log(`Submitting ${locationType} location:`, location);

        onSubmit({
            dataset: locationType,
            location,
            year,
            month
        });
    };

    return (
        <form onSubmit={handleSubmit}>
            <div className="form-group">
                <BlurText
                    text="Location Type:"
                    delay={150}
                    animateBy="words"
                    direction="top"
                    className="text-2xl lb-8"
                />
                <select
                    id="type"
                    value={locationType}
                    onChange={handleLocationTypeChange}
                    required
                >
                    <option value="country">Country</option>
                    <option value="state">State</option>
                    <option value="city">City</option>
                </select>
            </div>

            <div className="form-group">
                <BlurText
                    text="Location:"
                    delay={150}
                    animateBy="words"
                    direction="top"
                    className="text-2xl lb-8"
                />
                <select
                    id="location"
                    value={
                        locationType === 'country'
                            ? selectedLocation
                            : typeof selectedLocation === 'object'
                                ? JSON.stringify(selectedLocation)
                                : selectedLocation
                    }
                    onChange={handleLocationChange}
                    required
                    disabled={loadingLocations}
                >
                    <option value="">Select {locationType}</option>
                    {locations.map((loc, index) => {
                        if (locationType === 'country') {
                            return (
                                <option key={index} value={loc}>
                                    {String(loc)}
                                </option>
                            );
                        } else if (locationType === 'state') {
                            return (
                                <option key={index} value={JSON.stringify(loc)}>
                                    {String(loc.state ?? '')}, {String(loc.country ?? '')}
                                </option>
                            );
                        } else if (locationType === 'city') {
                            return (
                                <option key={index} value={JSON.stringify(loc)}>
                                    {String(loc.city ?? '')}, {String(loc.country ?? '')}
                                </option>
                            );
                        }
                        return null;
                    })}
                </select>
                {loadingLocations && <div className="spinner-small"></div>}
            </div>

            <div className="form-group">
                <BlurText
                    text="Year:"
                    delay={150}
                    animateBy="words"
                    direction="top"
                    className="text-2xl lb-8"
                />
                <input
                    type="number"
                    id="year"
                    value={year}
                    onChange={(e) => setYear(parseInt(e.target.value, 10))}
                    required
                />
            </div>

            <div className="form-group">
                <BlurText
                    text="Month:"
                    delay={150}
                    animateBy="words"
                    direction="top"
                    className="text-2xl lb-8"
                />
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
