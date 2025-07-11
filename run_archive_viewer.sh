#!/bin/bash

# Archive Viewer Launcher Script
# This script launches the Streamlit archive viewer application

echo "🚀 Starting Archive Viewer..."
echo "📊 Loading archived calls, eval runs, and judgement data..."
echo ""

# Check if we're in the right directory
if [ ! -f "archive_viewer.py" ]; then
    echo "❌ Error: archive_viewer.py not found in current directory"
    echo "Please run this script from the project root directory"
    exit 1
fi

# Check if required files exist
if [ ! -f "evals/archived_calls.jsonl" ]; then
    echo "⚠️  Warning: archived_calls.jsonl not found"
fi

if [ ! -f "evals/eval_run_calls.jsonl" ]; then
    echo "⚠️  Warning: eval_run_calls.jsonl not found"
fi

if [ ! -f "evals/judgement_results.jsonl" ]; then
    echo "⚠️  Warning: judgement_results.jsonl not found"
fi

echo "✅ Starting Streamlit application..."
echo "🌐 The application will open in your default web browser"
echo "📱 You can also access it at: http://localhost:8501"
echo ""

# Launch the Streamlit application
streamlit run archive_viewer.py 
