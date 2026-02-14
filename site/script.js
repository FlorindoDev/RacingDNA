/**
 * F1 Tempo Clone - Script.js
 * Handles UI logic, OpenF1 API integration, data caching, and Chart.js rendering.
 */

// ============================================================
//  DATA CONSTANTS
// ============================================================

const DRIVERS_2024 = [
    { code: 'VER', name: 'Max Verstappen', team: 'Red Bull Racing', color: '#3671C6', altColor: '#1e41ff' },
    { code: 'PER', name: 'Sergio Perez', team: 'Red Bull Racing', color: '#5b9bd5', altColor: '#6692FF' },
    { code: 'HAM', name: 'Lewis Hamilton', team: 'Mercedes', color: '#27F4D2', altColor: '#00d2be' },
    { code: 'RUS', name: 'George Russell', team: 'Mercedes', color: '#6CFCE6', altColor: '#27F4D2' },
    { code: 'LEC', name: 'Charles Leclerc', team: 'Ferrari', color: '#E80020', altColor: '#ff3333' },
    { code: 'SAI', name: 'Carlos Sainz', team: 'Ferrari', color: '#FF6666', altColor: '#ff8888' },
    { code: 'NOR', name: 'Lando Norris', team: 'McLaren', color: '#FF8000', altColor: '#ff9933' },
    { code: 'PIA', name: 'Oscar Piastri', team: 'McLaren', color: '#FFB366', altColor: '#ffcc66' },
    { code: 'ALO', name: 'Fernando Alonso', team: 'Aston Martin', color: '#229971', altColor: '#00665f' },
    { code: 'STR', name: 'Lance Stroll', team: 'Aston Martin', color: '#44cc99', altColor: '#2DB571' },
    { code: 'GAS', name: 'Pierre Gasly', team: 'Alpine', color: '#0093CC', altColor: '#00b8f1' },
    { code: 'OCO', name: 'Esteban Ocon', team: 'Alpine', color: '#55C8FF', altColor: '#70cbff' },
    { code: 'ALB', name: 'Alexander Albon', team: 'Williams', color: '#64C4FF', altColor: '#00a0dc' },
    { code: 'SAR', name: 'Logan Sargeant', team: 'Williams', color: '#99DDFF', altColor: '#37bedd' },
    { code: 'TSU', name: 'Yuki Tsunoda', team: 'RB', color: '#6692FF', altColor: '#4466dd' },
    { code: 'RIC', name: 'Daniel Ricciardo', team: 'RB', color: '#99B3FF', altColor: '#7788ff' },
    { code: 'BOT', name: 'Valtteri Bottas', team: 'Kick Sauber', color: '#52E252', altColor: '#00e000' },
    { code: 'ZHO', name: 'Guanyu Zhou', team: 'Kick Sauber', color: '#88FF88', altColor: '#55dd55' },
    { code: 'HUL', name: 'Nico Hulkenberg', team: 'Haas', color: '#B6BABD', altColor: '#999999' },
    { code: 'MAG', name: 'Kevin Magnussen', team: 'Haas', color: '#DDDDDD', altColor: '#cccccc' }
];

const RACES_2024 = [
    "Bahrain Grand Prix", "Saudi Arabian Grand Prix", "Australian Grand Prix",
    "Japanese Grand Prix", "Chinese Grand Prix", "Miami Grand Prix",
    "Emilia Romagna Grand Prix", "Monaco Grand Prix", "Canadian Grand Prix",
    "Spanish Grand Prix", "Austrian Grand Prix", "British Grand Prix",
    "Hungarian Grand Prix", "Belgian Grand Prix", "Dutch Grand Prix",
    "Italian Grand Prix", "Azerbaijan Grand Prix", "Singapore Grand Prix",
    "United States Grand Prix", "Mexico City Grand Prix", "São Paulo Grand Prix",
    "Las Vegas Grand Prix", "Qatar Grand Prix", "Abu Dhabi Grand Prix"
];

const RACES_2023 = RACES_2024; // Same calendar for simplicity

// ============================================================
//  APP STATE - with data caching
// ============================================================

const state = {
    selectedDrivers: ['VER', 'NOR'],
    selectedLap: 'fastest',
    sessionKey: null,
    dataSource: 'mock', // 'api' or 'mock'

    // Cached data per session – prevents regeneration on every click
    cache: {
        key: null, // `${year}-${race}-${session}`
        lapData: {},    // { driverCode: [...] }
        telemData: {},  // { driverCode: {...} }
        drivers: null   // array from API or fallback
    }
};

// ============================================================
//  CHART INSTANCES
// ============================================================

let lapChart, speedChart, throttleChart, brakeChart, gearChart;

// ============================================================
//  INITIALIZATION
// ============================================================

document.addEventListener('DOMContentLoaded', () => {
    initUI();
    initCharts();
    setupEventListeners();
    loadSession(); // Initial load with defaults
});

function initUI() {
    populateRaceSelect();
    populateDriverList();
}

function populateRaceSelect() {
    const year = document.getElementById('year-select').value;
    const raceSelect = document.getElementById('race-select');
    const races = year === '2023' ? RACES_2023 : RACES_2024;
    raceSelect.innerHTML = '';
    races.forEach((race, i) => {
        const option = document.createElement('option');
        option.value = race;
        option.text = race;
        raceSelect.add(option);
    });
}

function populateDriverList() {
    const list = document.getElementById('driver-list');
    list.innerHTML = '';

    const drivers = state.cache.drivers || DRIVERS_2024;

    drivers.forEach(driver => {
        const item = document.createElement('div');
        item.className = `driver-item ${state.selectedDrivers.includes(driver.code) ? 'selected' : ''}`;
        item.dataset.driver = driver.code;
        item.style.setProperty('--driver-color', driver.color);

        item.innerHTML = `
            <div class="checkbox-indicator"></div>
            <span class="color-dot"></span>
            <span class="driver-name">${driver.name}</span>
            <span class="driver-team">${driver.team}</span>
        `;

        item.addEventListener('click', () => toggleDriver(driver.code));
        list.appendChild(item);
    });
}

function toggleDriver(driverCode) {
    if (state.selectedDrivers.includes(driverCode)) {
        if (state.selectedDrivers.length > 1) {
            state.selectedDrivers = state.selectedDrivers.filter(d => d !== driverCode);
        }
    } else {
        if (state.selectedDrivers.length < 5) {
            state.selectedDrivers.push(driverCode);
        }
    }

    // Update sidebar UI
    document.querySelectorAll('.driver-item').forEach(el => {
        el.classList.toggle('selected', state.selectedDrivers.includes(el.dataset.driver));
    });

    // Update legend
    updateLegend();

    // Update charts with cached data (no regeneration)
    updateAllCharts();
}

function setupEventListeners() {
    // Load Session button
    document.getElementById('load-data-btn').addEventListener('click', () => loadSession());

    // Year change → refresh race list
    document.getElementById('year-select').addEventListener('change', () => {
        populateRaceSelect();
    });

    // Lap selector change
    document.getElementById('lap-select').addEventListener('change', (e) => {
        state.selectedLap = e.target.value;
        updateTelemetryCharts();
        updateLapInfo();
    });
}

// ============================================================
//  DATA LOADING
// ============================================================

async function loadSession() {
    const year = document.getElementById('year-select').value;
    const race = document.getElementById('race-select').value;
    const session = document.getElementById('session-select').value;
    const cacheKey = `${year}-${race}-${session}`;

    // If same session, don't reload
    if (state.cache.key === cacheKey) return;

    showLoading(true);
    setBtnLoading(true);

    // Try OpenF1 API first
    let success = false;
    try {
        success = await loadFromAPI(year, race, session);
    } catch (err) {
        console.warn('OpenF1 API failed, using mock data:', err);
    }

    if (!success) {
        loadMockData(year, race, session);
        state.dataSource = 'mock';
    } else {
        state.dataSource = 'api';
    }

    state.cache.key = cacheKey;
    state.selectedLap = 'fastest';

    // Populate lap selector
    populateLapSelector();
    updateLegend();
    updateAllCharts();
    updateSessionInfo();

    showLoading(false);
    setBtnLoading(false);
}

async function loadFromAPI(year, race, session) {
    // Step 1: Find the session key
    const sessionsUrl = `https://api.openf1.org/v1/sessions?year=${year}&session_name=${encodeURIComponent(session)}&country_name=${encodeURIComponent(getCountryFromRace(race))}`;
    const sessionsResp = await fetch(sessionsUrl);
    if (!sessionsResp.ok) throw new Error('Sessions API failed');

    const sessions = await sessionsResp.json();
    if (!sessions || sessions.length === 0) throw new Error('No session found');

    const sessionData = sessions[0];
    state.sessionKey = sessionData.session_key;

    // Step 2: Get drivers for this session
    const driversUrl = `https://api.openf1.org/v1/drivers?session_key=${state.sessionKey}`;
    const driversResp = await fetch(driversUrl);
    const apiDrivers = await driversResp.json();

    if (apiDrivers && apiDrivers.length > 0) {
        state.cache.drivers = apiDrivers.map(d => ({
            code: d.name_acronym || d.driver_number?.toString(),
            name: d.full_name || `${d.first_name} ${d.last_name}`,
            team: d.team_name || 'Unknown',
            color: '#' + (d.team_colour || '888888'),
            number: d.driver_number
        }));

        // Ensure distinct colors for same-team drivers
        differentiateTeamColors(state.cache.drivers);
        populateDriverList();
    }

    // Step 3: Load lap data for selected drivers
    for (const code of state.selectedDrivers) {
        await loadDriverLapsFromAPI(code);
    }

    return true;
}

async function loadDriverLapsFromAPI(driverCode) {
    if (!state.sessionKey) return;

    const driver = (state.cache.drivers || DRIVERS_2024).find(d => d.code === driverCode);
    if (!driver) return;

    const driverNum = driver.number;
    if (!driverNum) {
        // Fallback: generate mock data for this driver
        state.cache.lapData[driverCode] = generateLapData(driver);
        state.cache.telemData[driverCode] = generateTelemetry(driver);
        return;
    }

    try {
        // Lap times
        const lapsUrl = `https://api.openf1.org/v1/laps?session_key=${state.sessionKey}&driver_number=${driverNum}`;
        const lapsResp = await fetch(lapsUrl);
        const laps = await lapsResp.json();

        if (laps && laps.length > 0) {
            state.cache.lapData[driverCode] = laps.map(l => ({
                x: l.lap_number,
                y: l.lap_duration || 0,
                driver: driverCode,
                isPit: l.is_pit_out_lap || false,
                sector1: l.duration_sector_1,
                sector2: l.duration_sector_2,
                sector3: l.duration_sector_3
            })).filter(l => l.y > 0 && l.y < 200); // Filter out invalid laps
        } else {
            state.cache.lapData[driverCode] = generateLapData(driver);
        }

        // Telemetry (car data) – this can be very large, so we sample
        const telemUrl = `https://api.openf1.org/v1/car_data?session_key=${state.sessionKey}&driver_number=${driverNum}&speed>=0`;
        const telemResp = await fetch(telemUrl);
        const telem = await telemResp.json();

        if (telem && telem.length > 50) {
            // Sample every Nth point to keep it manageable
            const step = Math.max(1, Math.floor(telem.length / 300));
            const sampled = telem.filter((_, i) => i % step === 0);
            state.cache.telemData[driverCode] = sampled.map((t, i) => ({
                x: i,
                speed: t.speed || 0,
                throttle: t.throttle || 0,
                brake: t.brake || 0,
                gear: t.n_gear || 0,
                rpm: t.rpm || 0
            }));
        } else {
            state.cache.telemData[driverCode] = generateTelemetry(driver);
        }
    } catch (err) {
        console.warn(`Failed to load data for ${driverCode}:`, err);
        state.cache.lapData[driverCode] = generateLapData(driver);
        state.cache.telemData[driverCode] = generateTelemetry(driver);
    }
}

function loadMockData(year, race, session) {
    state.cache.drivers = [...DRIVERS_2024];
    state.cache.lapData = {};
    state.cache.telemData = {};

    const drivers = state.cache.drivers;
    drivers.forEach(driver => {
        state.cache.lapData[driver.code] = generateLapData(driver);
        state.cache.telemData[driver.code] = generateTelemetry(driver);
    });

    populateDriverList();
}

// ============================================================
//  MOCK DATA GENERATORS (fallback)
// ============================================================

function generateLapData(driver, laps = 55) {
    const data = [];
    // Base time varies by driver skill
    const driverIdx = DRIVERS_2024.findIndex(d => d.code === driver.code);
    let baseTime = 88 + (driverIdx >= 0 ? driverIdx * 0.08 : Math.random());

    let currentTire = 'S';
    let tireAge = 0;

    for (let i = 1; i <= laps; i++) {
        tireAge++;

        // Pit stop
        if (tireAge > 14 + Math.floor(Math.random() * 6)) {
            currentTire = currentTire === 'S' ? 'M' : (currentTire === 'M' ? 'H' : 'S');
            tireAge = 0;
            data.push({ x: i, y: baseTime + 18 + Math.random() * 4, tire: currentTire, driver: driver.code, isPit: true });
            continue;
        }

        // Degradation + variance
        let time = baseTime + (tireAge * 0.08) + (Math.random() * 0.6 - 0.3);

        // Tire performance offset
        if (currentTire === 'S') time -= 0.3;
        if (currentTire === 'H') time += 0.2;

        data.push({ x: i, y: time, tire: currentTire, driver: driver.code });
    }
    return data;
}

function generateTelemetry(driver, length = 300) {
    const data = [];
    let speed = 200;
    let throttle = 80;
    let brake = 0;
    let gear = 5;

    const seed = (DRIVERS_2024.findIndex(d => d.code === driver.code) + 1) * 7;

    for (let i = 0; i < length; i++) {
        const phase = (i + seed) % 60;

        if (phase < 35) {
            // Straight / accelerating
            throttle = Math.min(100, 60 + phase * 1.5 + (Math.random() - 0.5) * 5);
            brake = 0;
            speed += throttle * 0.03 - 0.5;
            gear = speed > 280 ? 8 : speed > 230 ? 7 : speed > 180 ? 6 : speed > 130 ? 5 : 4;
        } else if (phase < 45) {
            // Braking zone
            throttle = 0;
            brake = Math.min(100, 40 + (phase - 35) * 8);
            speed -= brake * 0.15;
            gear = speed > 200 ? 6 : speed > 150 ? 5 : speed > 100 ? 4 : 3;
        } else {
            // Corner / low speed
            throttle = Math.min(60, 10 + (phase - 45) * 4);
            brake = Math.max(0, 30 - (phase - 45) * 3);
            speed += (throttle - brake) * 0.05;
            gear = speed > 160 ? 5 : speed > 120 ? 4 : 3;
        }

        speed = Math.max(60, Math.min(340, speed + (Math.random() - 0.5) * 3));

        data.push({
            x: i * 15,
            speed: Math.round(speed),
            throttle: Math.round(Math.max(0, Math.min(100, throttle))),
            brake: Math.round(Math.max(0, Math.min(100, brake))),
            gear: Math.round(gear)
        });
    }
    return data;
}

// ============================================================
//  CHARTING
// ============================================================

function initCharts() {
    Chart.defaults.color = '#7a7a95';
    Chart.defaults.borderColor = '#252540';
    Chart.defaults.font.family = "'Titillium Web', sans-serif";
    Chart.defaults.font.size = 11;

    const tooltipStyle = {
        backgroundColor: 'rgba(15, 15, 30, 0.95)',
        titleColor: '#fff',
        bodyColor: '#ccc',
        borderColor: '#e10600',
        borderWidth: 1,
        cornerRadius: 4,
        padding: 10,
        titleFont: { weight: '700', size: 12 },
        bodyFont: { size: 11 },
        displayColors: true,
        boxWidth: 8,
        boxHeight: 8,
        callbacks: {
            labelColor: function (ctx) {
                return {
                    borderColor: ctx.dataset.borderColor,
                    backgroundColor: ctx.dataset.borderColor || ctx.dataset.backgroundColor,
                    borderRadius: 2
                };
            }
        }
    };

    const commonLineOptions = {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: 'index', intersect: false },
        plugins: {
            legend: { display: false },
            tooltip: { ...tooltipStyle }
        },
        scales: {
            x: { type: 'linear', display: false },
            y: { grid: { color: '#1f1f35', lineWidth: 0.5 } }
        },
        elements: { point: { radius: 0, hitRadius: 10 }, line: { borderWidth: 2 } },
        animation: { duration: 400 }
    };

    // 1. Lap Analysis Chart (Scatter)
    const ctxLap = document.getElementById('lapChart').getContext('2d');
    lapChart = new Chart(ctxLap, {
        type: 'scatter',
        data: { datasets: [] },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: { duration: 400 },
            scales: {
                x: {
                    type: 'linear', position: 'bottom',
                    title: { display: true, text: 'Lap Number', color: '#50506a', font: { size: 10 } },
                    grid: { color: '#1f1f35', lineWidth: 0.5 }
                },
                y: {
                    title: { display: true, text: 'Lap Time (s)', color: '#50506a', font: { size: 10 } },
                    grid: { color: '#1f1f35', lineWidth: 0.5 }
                }
            },
            plugins: {
                legend: { display: false },
                tooltip: {
                    ...tooltipStyle,
                    callbacks: {
                        title: (items) => `Lap ${items[0]?.raw?.x || ''}`,
                        label: (ctx) => {
                            const p = ctx.raw;
                            const timeStr = formatLapTime(p.y);
                            const tire = p.tire ? ` · ${p.tire}` : '';
                            const pit = p.isPit ? ' ⟳ PIT' : '';
                            return ` ${p.driver}: ${timeStr}${tire}${pit}`;
                        },
                        labelColor: (ctx) => ({
                            borderColor: ctx.dataset.borderColor,
                            backgroundColor: ctx.dataset.borderColor,
                            borderRadius: 2
                        })
                    }
                }
            },
            onClick: (e, elements) => {
                if (elements.length > 0) {
                    const idx = elements[0].index;
                    const dsIdx = elements[0].datasetIndex;
                    const point = lapChart.data.datasets[dsIdx].data[idx];
                    const lapNum = point.x;

                    // Set selected lap in dropdown
                    const lapSelect = document.getElementById('lap-select');
                    lapSelect.value = lapNum.toString();
                    state.selectedLap = lapNum.toString();

                    updateTelemetryCharts();
                    updateLapInfo();
                }
            }
        }
    });

    // 2. Speed Chart
    speedChart = new Chart(document.getElementById('speedChart').getContext('2d'), {
        type: 'line', data: { datasets: [] },
        options: {
            ...commonLineOptions,
            plugins: {
                ...commonLineOptions.plugins,
                tooltip: {
                    ...tooltipStyle,
                    callbacks: {
                        ...tooltipStyle.callbacks,
                        label: (ctx) => ` ${ctx.dataset.label}: ${ctx.parsed.y} km/h`
                    }
                }
            }
        }
    });

    // 3. Throttle Chart
    throttleChart = new Chart(document.getElementById('throttleChart').getContext('2d'), {
        type: 'line', data: { datasets: [] },
        options: {
            ...commonLineOptions,
            scales: { ...commonLineOptions.scales, y: { ...commonLineOptions.scales.y, min: 0, max: 105 } }
        }
    });

    // 4. Brake Chart
    brakeChart = new Chart(document.getElementById('brakeChart').getContext('2d'), {
        type: 'line', data: { datasets: [] },
        options: {
            ...commonLineOptions,
            scales: { ...commonLineOptions.scales, y: { ...commonLineOptions.scales.y, min: 0, max: 105 } }
        }
    });

    // 5. Gear Chart
    gearChart = new Chart(document.getElementById('gearChart').getContext('2d'), {
        type: 'line', data: { datasets: [] },
        options: {
            ...commonLineOptions,
            scales: {
                ...commonLineOptions.scales,
                y: { ...commonLineOptions.scales.y, min: 0, max: 9, ticks: { stepSize: 1 } }
            },
            elements: { ...commonLineOptions.elements, line: { borderWidth: 2, stepped: 'before' } }
        }
    });
}

// ============================================================
//  CHART UPDATES
// ============================================================

function updateAllCharts() {
    updateLapChart();
    updateTelemetryCharts();
}

function updateLapChart() {
    const drivers = state.cache.drivers || DRIVERS_2024;

    const datasets = state.selectedDrivers.map(code => {
        const driver = drivers.find(d => d.code === code);
        if (!driver) return null;

        // Ensure data exists in cache
        if (!state.cache.lapData[code]) {
            state.cache.lapData[code] = generateLapData(driver);
        }
        const lapData = state.cache.lapData[code];

        return {
            label: driver.name,
            data: lapData,
            backgroundColor: driver.color,
            borderColor: driver.color,
            pointRadius: 4,
            pointHoverRadius: 7,
            pointBorderWidth: 0,
            pointHoverBorderWidth: 2,
            pointHoverBorderColor: '#fff'
        };
    }).filter(Boolean);

    lapChart.data.datasets = datasets;
    lapChart.update();
}

function updateTelemetryCharts() {
    const drivers = state.cache.drivers || DRIVERS_2024;
    const datasetsSpeed = [];
    const datasetsThrottle = [];
    const datasetsBrake = [];
    const datasetsGear = [];

    state.selectedDrivers.forEach(code => {
        const driver = drivers.find(d => d.code === code);
        if (!driver) return;

        // Ensure telemetry exists in cache
        if (!state.cache.telemData[code]) {
            state.cache.telemData[code] = generateTelemetry(driver);
        }
        const telemData = state.cache.telemData[code];

        const commonDs = {
            label: driver.name,
            borderColor: driver.color,
            borderWidth: 1.5,
            tension: 0.2,
            pointRadius: 0,
            fill: false
        };

        datasetsSpeed.push({ ...commonDs, data: telemData.map(d => ({ x: d.x, y: d.speed })) });
        datasetsThrottle.push({ ...commonDs, data: telemData.map(d => ({ x: d.x, y: d.throttle })) });
        datasetsBrake.push({ ...commonDs, data: telemData.map(d => ({ x: d.x, y: d.brake })) });
        datasetsGear.push({
            ...commonDs,
            data: telemData.map(d => ({ x: d.x, y: d.gear })),
            stepped: 'before'
        });
    });

    speedChart.data.datasets = datasetsSpeed;
    speedChart.update();

    throttleChart.data.datasets = datasetsThrottle;
    throttleChart.update();

    brakeChart.data.datasets = datasetsBrake;
    brakeChart.update();

    gearChart.data.datasets = datasetsGear;
    gearChart.update();
}

// ============================================================
//  LEGEND & LAP SELECTOR
// ============================================================

function updateLegend() {
    const legend = document.getElementById('lap-legend');
    const drivers = state.cache.drivers || DRIVERS_2024;
    legend.innerHTML = '';

    state.selectedDrivers.forEach(code => {
        const driver = drivers.find(d => d.code === code);
        if (!driver) return;
        const item = document.createElement('div');
        item.className = 'legend-item';
        item.innerHTML = `<span class="legend-dot" style="background:${driver.color}; box-shadow: 0 0 6px ${driver.color}55;"></span>${driver.code}`;
        legend.appendChild(item);
    });
}

function populateLapSelector() {
    const select = document.getElementById('lap-select');
    select.innerHTML = '<option value="fastest">Fastest Lap</option>';

    // Determine max laps from cached data
    let maxLaps = 0;
    Object.values(state.cache.lapData).forEach(laps => {
        if (laps && laps.length > maxLaps) maxLaps = laps.length;
    });

    for (let i = 1; i <= maxLaps; i++) {
        const option = document.createElement('option');
        option.value = i;
        option.text = `Lap ${i}`;
        select.add(option);
    }
}

function updateLapInfo() {
    const info = document.getElementById('selected-lap-info');
    if (state.selectedLap === 'fastest') {
        info.textContent = 'Showing fastest lap telemetry';
    } else {
        info.textContent = `Showing telemetry for Lap ${state.selectedLap}`;
    }
}

// ============================================================
//  UI HELPERS
// ============================================================

function showLoading(show) {
    const overlay = document.getElementById('loading-overlay');
    overlay.style.display = show ? 'flex' : 'none';
}

function setBtnLoading(loading) {
    const btn = document.getElementById('load-data-btn');
    const text = btn.querySelector('.btn-text');
    const loader = btn.querySelector('.btn-loader');
    if (loading) {
        text.textContent = 'Loading';
        loader.style.display = 'block';
        btn.disabled = true;
    } else {
        text.textContent = 'Load Session';
        loader.style.display = 'none';
        btn.disabled = false;
    }
}

function updateSessionInfo() {
    const info = document.getElementById('session-info');
    const badge = state.dataSource === 'api'
        ? '<span class="status-badge live">LIVE DATA</span>'
        : '<span class="status-badge mock">MOCK DATA</span>';
    info.innerHTML = badge;
}

function formatLapTime(seconds) {
    if (!seconds || seconds <= 0) return '--:--.---';
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toFixed(3).padStart(6, '0')}`;
}

function getCountryFromRace(raceName) {
    const map = {
        'Bahrain Grand Prix': 'Bahrain',
        'Saudi Arabian Grand Prix': 'Saudi Arabia',
        'Australian Grand Prix': 'Australia',
        'Japanese Grand Prix': 'Japan',
        'Chinese Grand Prix': 'China',
        'Miami Grand Prix': 'United States',
        'Emilia Romagna Grand Prix': 'Italy',
        'Monaco Grand Prix': 'Monaco',
        'Canadian Grand Prix': 'Canada',
        'Spanish Grand Prix': 'Spain',
        'Austrian Grand Prix': 'Austria',
        'British Grand Prix': 'United Kingdom',
        'Hungarian Grand Prix': 'Hungary',
        'Belgian Grand Prix': 'Belgium',
        'Dutch Grand Prix': 'Netherlands',
        'Italian Grand Prix': 'Italy',
        'Azerbaijan Grand Prix': 'Azerbaijan',
        'Singapore Grand Prix': 'Singapore',
        'United States Grand Prix': 'United States',
        'Mexico City Grand Prix': 'Mexico',
        'São Paulo Grand Prix': 'Brazil',
        'Las Vegas Grand Prix': 'United States',
        'Qatar Grand Prix': 'Qatar',
        'Abu Dhabi Grand Prix': 'United Arab Emirates'
    };
    return map[raceName] || raceName.replace(' Grand Prix', '');
}

function differentiateTeamColors(drivers) {
    const teamCount = {};
    drivers.forEach(d => {
        const team = d.team;
        if (!teamCount[team]) teamCount[team] = [];
        teamCount[team].push(d);
    });

    Object.values(teamCount).forEach(teamDrivers => {
        if (teamDrivers.length > 1) {
            // Lighten color for the 2nd driver
            for (let i = 1; i < teamDrivers.length; i++) {
                teamDrivers[i].color = lightenColor(teamDrivers[0].color, 30 + i * 15);
            }
        }
    });
}

function lightenColor(hex, amount) {
    hex = hex.replace('#', '');
    if (hex.length === 3) hex = hex.split('').map(c => c + c).join('');
    const r = Math.min(255, parseInt(hex.substring(0, 2), 16) + amount);
    const g = Math.min(255, parseInt(hex.substring(2, 4), 16) + amount);
    const b = Math.min(255, parseInt(hex.substring(4, 6), 16) + amount);
    return `#${r.toString(16).padStart(2, '0')}${g.toString(16).padStart(2, '0')}${b.toString(16).padStart(2, '0')}`;
}
