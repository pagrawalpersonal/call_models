# Archive Viewer

A Streamlit-based tool for viewing and analyzing archived API calls, eval runs, and judgement results.

## Features

- **📊 Tabular Display**: View archived calls in a sortable, searchable table
- **🔍 Search Functionality**: Search across all fields in the archive data
- **📄 Server-side Pagination**: Handle large datasets efficiently
- **🔗 Relationship Visualization**: View connected eval runs and judgements
- **📋 Detailed Views**: Expand entries to see full details
- **📈 Statistics**: View summary statistics in the sidebar

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements_streamlit.txt
```

## Usage

1. Run the Streamlit application:
```bash
streamlit run archive_viewer.py
```

2. The application will open in your default web browser

## Data Structure

The application reads from three JSONL files:

- `call_models/evals/archived_calls.jsonl`: Main archive data
- `call_models/evals/eval_run_calls.jsonl`: Eval run data linked to archives
- `call_models/evals/judgement_results.jsonl`: Judgement data linked to archives

## Features Explained

### Search and Filter
- **Search**: Enter text to search across all fields
- **Model Filter**: Filter by specific model names
- **Tag Filter**: Filter by specific tags

### Pagination
- Configure items per page (10, 25, 50, 100)
- Navigate through pages of results

### Detailed View
- Click on any entry to see full details
- View system prompts, user prompts, and responses
- See connected eval runs and judgements
- View debug information and metadata

### Relationships
- **Eval Runs**: Shows when an archive entry has been evaluated with different models
- **Judgements**: Shows when responses have been judged for quality comparison

## Data Fields

### Archive Entries
- **ID**: Unique identifier for the archive entry
- **Timestamp**: When the call was made
- **Model**: The model used for the call
- **Tag**: Optional tag for categorization
- **Time Taken**: Response time in seconds
- **Input/Output Tokens**: Token usage (if available)
- **System/User Prompts**: The prompts used
- **Response**: The model's response
- **Has Eval Runs**: Whether this entry has eval runs
- **Has Judgements**: Whether this entry has judgements

### Eval Runs
- **Eval Run ID**: Unique identifier for the eval run
- **Model Used**: The model used for evaluation
- **Original Model**: The model from the original archive
- **Response**: The evaluation response
- **Debug Info**: Analysis of prompt difficulties and improvements

### Judgements
- **Model New/Original**: The models being compared
- **Judge Score**: Quality score (-1 to 1)
- **Judge Reasoning**: Detailed reasoning for the score
- **Judge Improvements**: Suggested improvements
- **Original/New Response**: The responses being compared

## File Structure

```
reddit_fetcher/
├── archive_viewer.py              # Main Streamlit application
├── requirements_streamlit.txt     # Python dependencies
├── README_archive_viewer.md      # This file
└── call_models/evals/
    ├── archived_calls.jsonl      # Archive data
    ├── eval_run_calls.jsonl      # Eval run data
    └── judgement_results.jsonl   # Judgement data
```

## Troubleshooting

1. **File not found errors**: Ensure the JSONL files exist in the correct paths
2. **Large file performance**: Use pagination to handle large datasets
3. **Search not working**: Check that the search term exists in the data

## Customization

You can modify the application by:

1. **Adding new filters**: Edit the sidebar section in `archive_viewer.py`
2. **Changing display fields**: Modify the `create_archive_dataframe` method
3. **Adding new visualizations**: Add new sections to the main display logic
4. **Customizing styling**: Modify the CSS in the `st.markdown` section 