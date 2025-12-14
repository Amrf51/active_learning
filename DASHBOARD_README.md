# Active Learning Dashboard

This interactive dashboard replaces the static `app.py` with a full multi-page interface for controlling and monitoring Active Learning experiments.

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Start the dashboard:**
   ```bash
   streamlit run dashboard.py
   ```

3. **Navigate to the Configuration page** to set up your first experiment

4. **Start the worker process** (displayed after experiment creation):
   ```bash
   python run_worker.py --experiment-id <your_experiment_id>
   ```

5. **Use the Active Learning page** to control training and monitor progress

## Pages Overview

### 🏠 Main Dashboard
- Experiment selection and overview
- Quick stats for selected experiments
- Recent experiments list

### ⚙️ Configuration
- **Model Selection:** ResNet-18, ResNet-50, MobileNetV2
- **Sampling Strategy:** Uncertainty, Entropy, Margin, Random
- **Training Parameters:** Cycles, pool sizes, epochs, batch size
- **Reset Mode:** Pretrained, head_only, none
- **Dataset Information:** Class distribution and statistics
- **Experiment Initialization:** Creates new experiments with validation

### 🎯 Active Learning (Coming Next)
- Real-time training visualization
- Live loss and accuracy curves
- Prediction monitor for reference images
- Query visualization and annotation interface
- Cycle control (Start, Pause, Stop)

### 📊 Results (Coming Next)
- Performance comparison across cycles
- Confusion matrix visualization
- Per-class metrics analysis
- Multi-experiment comparison
- Results export functionality

## Architecture

The dashboard follows the **Controller-Worker Pattern**:

- **Dashboard (Controller):** Reads state, writes commands, displays visualizations
- **Worker Process:** Executes training, updates state, handles heavy computation
- **Communication:** Via `ExperimentState.json` with FileLock for atomic operations

## Key Features Implemented

✅ **Multi-page Streamlit app structure**
✅ **Experiment management and selection**
✅ **Comprehensive configuration interface**
✅ **Form validation and error handling**
✅ **Dataset information display**
✅ **Experiment initialization with state management**
✅ **Integration with existing backend (StateManager, Config)**

## Next Steps

The remaining tasks will implement:
- Active Learning control page with live training visualization
- Query visualizer and annotation interface  
- Results analysis and comparison tools
- Error handling and recovery options

## File Structure

```
al-car-classification/
├── dashboard.py              # Main dashboard entry point
├── pages/
│   └── 1_⚙️_Configuration.py # Configuration page
├── src/
│   ├── state.py             # State management (existing)
│   ├── config.py            # Configuration classes (existing)
│   └── ...                  # Other backend modules
└── experiments/             # Experiment data directory
```

## Usage Tips

1. **Experiment Names:** Use descriptive names like `uncertainty_resnet18_baseline`
2. **Model Selection:** Start with ResNet-18 for faster experimentation
3. **Pool Sizes:** Begin with small pools (50-100 initial, 10-20 query batch)
4. **Reset Mode:** Use "pretrained" for fair cycle comparisons
5. **Validation:** Pay attention to configuration warnings and errors

## Troubleshooting

- **Import Errors:** Ensure you're in the correct directory and dependencies are installed
- **State File Issues:** Check that the `experiments/` directory has proper permissions
- **Worker Connection:** Verify the worker process is running with the correct experiment ID