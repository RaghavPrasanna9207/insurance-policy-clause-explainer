import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter, Route, Routes } from 'react-router-dom'

import './index.css'
import { Policy } from './routes/Policy'
import { Upload } from './routes/Upload'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Upload />} />
        {/* The document id is in the URL so a long analysis survives a reload
            and the result is a link the reader can come back to. */}
        <Route path="/policy/:id" element={<Policy />} />
      </Routes>
    </BrowserRouter>
  </StrictMode>,
)
