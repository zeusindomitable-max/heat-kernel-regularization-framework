# Heat-Kernel Regularization Framework (v0.9)

A research-grade framework implementing variational heat-kernel regularization
for curvature-aware learning and geometric optimization.

---

## 🔬 Core Features
- Variational functional with task, Ricci, and curvature regularization terms.
- Heat-kernel smoothing with Seeley–DeWitt subtraction (UV regularization).
- Hutchinson trace estimator for Hessian-based curvature.
- Modular controller field τ(θ, t) with gradient-based updates.
- Clean autograd pipeline using PyTorch.

---

## 📦 Requirements

```bash
pip install -r requirements.txt
```
Main dependencies:

•torch>=2.1

•numpy

•tqdm

•matplotlib

•pytest

▶️ Running Demo
```bash
python main.py
```
Or open demo.ipynb to visualize curvature and loss dynamics

🧪 Testing
```bash
pytest tests/
```
All tests should pass cleanly with fresh environment.



⚖️ License
```bash

---

### 📄 **LICENSE (MIT)**

```text
MIT License

Copyright (c) 2025 [H. Hardiyan]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the “Software”), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

