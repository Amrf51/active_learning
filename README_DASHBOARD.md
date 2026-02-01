# Active Learning Dashboard - Quick Start

## Running the Dashboard

### Option 1: Using the launcher script (Recommended)
```bash
cd al-car-classification
python run_dashboard.py
```

### Option 2: Direct streamlit command
```bash
cd al-car-classification
streamlit run dashboard.py
```

### Option 3: Test imports first
```bash
cd al-car-classification
python test_imports.py
```

## Troubleshooting

### "ModuleNotFoundError: No module named 'views'"

**Solution:** Make sure you're running from the `al-car-classification` directory:
```bash
cd al-car-classification  # Make sure you're in this directory
python test_imports.py    # Test imports first
streamlit run dashboard.py
```

### Import errors for controller/model modules

**Check:**
1. All `__init__.py` files exist in subdirectories
2. You're in the correct directory (`al-car-classification`)
3. Python can find the modules

### Database errors

The dashboard will automatically create a SQLite database at:
`al-car-classification/experiments/al_dashboard.db`

## File Structure

```
al-car-classification/
├── dashboard.py              # Main entry point
├── run_dashboard.py         # Launcher script
├── test_imports.py          # Import test script
├── views/
│   ├── __init__.py
│   ├── controller_factory.py
│   ├── 1_Configuration.py
│   ├── 2_Active_Learning.py
│   └── 3_Results.py
├── controller/
│   ├── __init__.py
│   ├── dispatcher.py
│   ├── model_handler.py
│   └── ...
├── model/
│   ├── __init__.py
│   ├── schemas.py
│   ├── world_state.py
│   └── database.py
└── ...
```

## Next Steps

1. **Test the dashboard startup** - Run `python test_imports.py`
2. **Start the dashboard** - Run `python run_dashboard.py`
3. **Create your first experiment** - Go to the Configuration page
4. **Monitor training** - Use the Active Learning page
5. **Analyze results** - Check the Results page