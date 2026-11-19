import { useState } from 'react'
import './App.css'

const initialForm = {
  age: '',
  gender: '2',
  height: '',
  weight: '',
  ap_hi: '',
  ap_lo: '',
  cholesterol: '1',
  gluc: '1',
  smoke: '0',
  alco: '0',
  active: '1',
}

function App() {
  const [form, setForm] = useState(initialForm)
  const [submitted, setSubmitted] = useState(null)

  const handleChange = (e) => {
    setForm({ ...form, [e.target.name]: e.target.value })
  }

  const handleSubmit = (e) => {
    e.preventDefault()
    setSubmitted(form)
    console.log('Form data:', form)
  }

  return (
    <div className="container">
      <h1>Cardiovascular Risk Digital Twin</h1>
      <p className="subtitle">Enter patient details to estimate risk.</p>

      <form onSubmit={handleSubmit}>
        <label>
          Age (years)
          <input type="number" name="age" min="1" max="120"
            value={form.age} onChange={handleChange} required />
        </label>

        <label>
          Gender
          <select name="gender" value={form.gender} onChange={handleChange}>
            <option value="2">Male</option>
            <option value="1">Female</option>
          </select>
        </label>

        <label>
          Height (cm)
          <input type="number" name="height" min="100" max="220"
            value={form.height} onChange={handleChange} required />
        </label>

        <label>
          Weight (kg)
          <input type="number" name="weight" min="30" max="200"
            value={form.weight} onChange={handleChange} required />
        </label>

        <label>
          Systolic BP (ap_hi)
          <input type="number" name="ap_hi" min="70" max="250"
            value={form.ap_hi} onChange={handleChange} required />
        </label>

        <label>
          Diastolic BP (ap_lo)
          <input type="number" name="ap_lo" min="40" max="150"
            value={form.ap_lo} onChange={handleChange} required />
        </label>

        <label>
          Cholesterol
          <select name="cholesterol" value={form.cholesterol} onChange={handleChange}>
            <option value="1">Normal</option>
            <option value="2">Above normal</option>
            <option value="3">Well above normal</option>
          </select>
        </label>

        <label>
          Glucose
          <select name="gluc" value={form.gluc} onChange={handleChange}>
            <option value="1">Normal</option>
            <option value="2">Above normal</option>
            <option value="3">Well above normal</option>
          </select>
        </label>

        <label>
          Smoker
          <select name="smoke" value={form.smoke} onChange={handleChange}>
            <option value="0">No</option>
            <option value="1">Yes</option>
          </select>
        </label>

        <label>
          Alcohol intake
          <select name="alco" value={form.alco} onChange={handleChange}>
            <option value="0">No</option>
            <option value="1">Yes</option>
          </select>
        </label>

        <label>
          Physically active
          <select name="active" value={form.active} onChange={handleChange}>
            <option value="1">Yes</option>
            <option value="0">No</option>
          </select>
        </label>

        <button type="submit">Predict Risk</button>
      </form>

      {submitted && (
        <div className="result">
          <h2>Submitted data</h2>
          <pre>{JSON.stringify(submitted, null, 2)}</pre>
        </div>
      )}
    </div>
  )
}

export default App