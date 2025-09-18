const API_BASE = location.hostname === 'localhost' ? 'http://localhost:8000' : 'http://localhost:8000';

const input = document.getElementById('image-input');
const nameInput = document.getElementById('name-input');
const form = document.getElementById('upload-form');
const preview = document.getElementById('preview');
const previewImg = document.getElementById('preview-img');
const result = document.getElementById('result');

input.addEventListener('change', () => {
  const file = input.files?.[0];
  if (!file) return;
  const url = URL.createObjectURL(file);
  previewImg.src = url;
  preview.classList.remove('hidden');
});

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const file = input.files?.[0];
  if (!file) { alert('Please choose an image first.'); return; }

  result.classList.remove('hidden');
  result.innerHTML = '<small class="muted">Analyzing…</small>';
    // Preload known foods for suggestions
    fetch(`${API_BASE}/foods`).then(r=>r.json()).then(d=>{
      const list = document.getElementById('foods-list');
      if (!list || !d || !Array.isArray(d.foods)) return;
      list.innerHTML = d.foods.map(f=>`<option value="${f}"></option>`).join('');
    }).catch(()=>{});

  const formData = new FormData();
  formData.append('image', file, file.name);
  if (nameInput.value.trim()) formData.append('name', nameInput.value.trim());

  try {
    const res = await fetch(`${API_BASE}/analyze`, { method: 'POST', body: formData });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Error ${res.status}`);
    }
    const data = await res.json();
    renderResult(data);
  } catch (err) {
      const tip = `Tip: Type the food name in the field above (e.g., banana) if the image cannot be auto-identified.`;
      result.innerHTML = `<div class="kv"><h4>Error</h4><div>${(err && err.message) || 'Unknown error'}</div><div><small class="muted">${tip}</small></div></div>`;
  }
});

function renderResult(data){
  const { item, confidence, nutrition, ayurveda, source } = data;
  const conf = confidence ? ` ${(confidence*100).toFixed(1)}%` : '';

  const ayuHtml = ayurveda ? `
    <div class="kv">
      <h4>Ayurvedic Insights</h4>
      <div><b>Rasa:</b> ${(ayurveda.rasa||[]).join(', ')}</div>
      <div><b>Virya:</b> ${ayurveda.virya||'-'}</div>
      <div><b>Vipaka:</b> ${ayurveda.vipaka||'-'}</div>
      <div><b>Doshas:</b> Vata: ${ayurveda.dosha_effect?.vata}, Pitta: ${ayurveda.dosha_effect?.pitta}, Kapha: ${ayurveda.dosha_effect?.kapha}</div>
      <div><b>Gunas:</b> ${(ayurveda.gunas||[]).join(', ')}</div>
      <div><b>Notes:</b> ${ayurveda.notes||''}</div>
    </div>
  ` : '';

  const nutrHtml = nutrition ? `
    <div class="kv">
      <h4>Nutrition</h4>
      ${nutrition.description ? `<div><b>Food:</b> ${nutrition.description}</div>` : ''}
      ${nutrition.brand ? `<div><b>Brand:</b> ${nutrition.brand}</div>` : ''}
      ${nutrition.serving ? `<div><b>Serving:</b> ${nutrition.serving}</div>` : ''}
      ${nutrition.calories ? `<div><b>Calories:</b> ${nutrition.calories}</div>` : ''}
      ${nutrition.macros ? `<div><b>Macros:</b> C:${nutrition.macros.carbs_g}g P:${nutrition.macros.protein_g}g F:${nutrition.macros.fat_g}g</div>` : ''}
      ${nutrition.micros ? `<div><b>Micros:</b> ${Object.entries(nutrition.micros).map(([k,v])=>`${k}: ${v}`).join(', ')}</div>` : ''}
      ${nutrition.nutrients ? `<div><b>Nutrients:</b> ${Object.entries(nutrition.nutrients).slice(0,10).map(([k,v])=>`${k}: ${v}`).join(', ')}</div>` : ''}
      <div><small class="muted">Source: ${nutrition.source||'unknown'}</small></div>
    </div>
  ` : '';

  result.innerHTML = `
    <div class="grid">
      <div class="kv">
        <h4>Prediction</h4>
        <div><b>Item:</b> ${item}${conf}</div>
        <div><small class="muted">Classifier: ${source.classifier || 'none'}</small></div>
        ${source.classifier_error ? `<div><small class="muted">Model error: ${source.classifier_error}</small></div>` : ''}
      </div>
      ${nutrHtml}
      ${ayuHtml}
    </div>
  `;
}
