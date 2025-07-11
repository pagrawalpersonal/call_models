import streamlit as st
import json
import pandas as pd
from datetime import datetime
import os
from typing import Dict, List, Optional, Any
import uuid

# Page configuration
st.set_page_config(
    page_title="Archive Viewer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .relationship-card {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .json-display {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        font-family: monospace;
        font-size: 0.8rem;
        max-height: 300px;
        overflow-y: auto;
    }
</style>
""", unsafe_allow_html=True)

class ArchiveViewer:
    def __init__(self):
        self.archived_calls_file = "evals/archived_calls.jsonl"
        self.eval_runs_file = "evals/eval_run_calls.jsonl"
        self.judgement_results_file = "evals/judgement_results.jsonl"
        
        # Load data
        self.archived_calls = self.load_jsonl(self.archived_calls_file)
        self.eval_runs = self.load_jsonl(self.eval_runs_file)
        self.judgement_results = self.load_jsonl(self.judgement_results_file)
        
        # Create relationships
        self.create_relationships()
    
    def load_jsonl(self, filepath: str) -> List[Dict]:
        """Load JSONL file and return list of dictionaries"""
        if not os.path.exists(filepath):
            st.error(f"File not found: {filepath}")
            return []
        
        data = []
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
        except Exception as e:
            st.error(f"Error loading {filepath}: {str(e)}")
            return []
        
        return data
    
    def create_relationships(self):
        """Create relationships between archives, eval runs, and judgements"""
        # Create lookup dictionaries
        self.eval_runs_by_archive = {}
        self.judgements_by_archive = {}
        
        # Group eval runs by archive_id
        for eval_run in self.eval_runs:
            archive_id = eval_run.get('archive_id')
            if archive_id:
                if archive_id not in self.eval_runs_by_archive:
                    self.eval_runs_by_archive[archive_id] = []
                self.eval_runs_by_archive[archive_id].append(eval_run)
        
        # Group judgements by archive_id
        for judgement in self.judgement_results:
            archive_id = judgement.get('archive_id')
            if archive_id:
                if archive_id not in self.judgements_by_archive:
                    self.judgements_by_archive[archive_id] = []
                self.judgements_by_archive[archive_id].append(judgement)
    
    def format_timestamp(self, timestamp_str: str) -> str:
        """Format timestamp for display"""
        try:
            dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            return dt.strftime('%Y-%m-%d %H:%M:%S')
        except:
            return timestamp_str
    
    def truncate_text(self, text: str, max_length: int = 100) -> str:
        """Truncate text for display"""
        if not text:
            return ""
        if len(text) <= max_length:
            return text
        return text[:max_length] + "..."
    
    def search_data(self, data: List[Dict], search_term: str) -> List[Dict]:
        """Search through data for matching terms"""
        if not search_term:
            return data
        
        search_term = search_term.lower()
        filtered_data = []
        
        for item in data:
            # Convert item to string for searching
            item_str = json.dumps(item, default=str).lower()
            if search_term in item_str:
                filtered_data.append(item)
        
        return filtered_data
    
    def create_archive_dataframe(self, data: List[Dict]) -> pd.DataFrame:
        """Create a pandas DataFrame from archive data"""
        if not data:
            return pd.DataFrame()
        
        # Prepare data for DataFrame
        df_data = []
        for item in data:
            df_data.append({
                'ID': item.get('id', ''),
                'Timestamp': self.format_timestamp(item.get('timestamp', '')),
                'Model': item.get('model', ''),
                'Tag': item.get('tag', ''),
                'Time Taken (s)': round(item.get('time_taken', 0), 3),
                'Input Tokens': item.get('input_tokens', ''),
                'Output Tokens': item.get('output_tokens', ''),
                'System Prompt': self.truncate_text(item.get('system_prompt_template', ''), 50),
                'User Prompt': self.truncate_text(item.get('user_prompt_template', ''), 50),
                'Response': self.truncate_text(str(item.get('response', '')), 50),
                'Has Eval Runs': 'Yes' if item.get('id') in self.eval_runs_by_archive else 'No',
                'Has Judgements': 'Yes' if item.get('id') in self.judgements_by_archive else 'No',
                'Raw Data': item  # Keep raw data for detailed view
            })
        
        return pd.DataFrame(df_data)
    
    def display_archive_details(self, archive_data: Dict):
        """Display detailed information about an archive entry"""
        st.subheader("📋 Archive Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("ID", archive_data.get('id', ''))
            st.metric("Model", archive_data.get('model', ''))
            st.metric("Time Taken", f"{archive_data.get('time_taken', 0):.3f}s")
            st.metric("Tag", archive_data.get('tag', 'None'))
        
        with col2:
            st.metric("Timestamp", self.format_timestamp(archive_data.get('timestamp', '')))
            st.metric("Input Tokens", archive_data.get('input_tokens', 'N/A'))
            st.metric("Output Tokens", archive_data.get('output_tokens', 'N/A'))
        
        # System and User Prompts
        st.subheader("💬 Prompts")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**System Prompt:**")
            st.code(archive_data.get('system_prompt_template', ''), language='text')
        
        with col2:
            st.write("**User Prompt:**")
            st.code(archive_data.get('user_prompt_template', ''), language='text')
        
        # Response
        st.subheader("📤 Response")
        response = archive_data.get('response', '')
        if isinstance(response, dict):
            st.json(response)
        else:
            st.code(str(response), language='json')
        
        # Debug Info
        debug_info = archive_data.get('debug_info')
        if debug_info:
            st.subheader("🐛 Debug Information")
            col1, col2 = st.columns(2)
            
            with col1:
                if debug_info.get('prompt_difficulties'):
                    st.write("**Prompt Difficulties:**")
                    st.info(debug_info['prompt_difficulties'])
            
            with col2:
                if debug_info.get('prompt_improvements'):
                    st.write("**Prompt Improvements:**")
                    st.success(debug_info['prompt_improvements'])
    
    def display_eval_runs(self, archive_id: str):
        """Display eval runs for an archive entry"""
        eval_runs = self.eval_runs_by_archive.get(archive_id, [])
        
        if not eval_runs:
            st.info("No eval runs found for this archive entry.")
            return
        
        st.subheader(f"🔄 Eval Runs ({len(eval_runs)})")
        
        for i, eval_run in enumerate(eval_runs):
            with st.expander(f"Eval Run {i+1}: {eval_run.get('model_used', 'Unknown Model')}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Eval Run ID:**", eval_run.get('eval_run_id', ''))
                    st.write("**Model Used:**", eval_run.get('model_used', ''))
                    st.write("**Original Model:**", eval_run.get('model_original', ''))
                    st.write("**DateTime:**", self.format_timestamp(eval_run.get('datetime', '')))
                
                with col2:
                    st.write("**Response:**")
                    response = eval_run.get('response', {})
                    if isinstance(response, dict):
                        st.json(response)
                    else:
                        st.code(str(response), language='json')
                
                # Debug info for eval run
                debug_info = eval_run.get('debug_info', {})
                if debug_info:
                    st.write("**Debug Info:**")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if debug_info.get('prompt_difficulties'):
                            st.write("**Prompt Difficulties:**")
                            st.info(debug_info['prompt_difficulties'])
                    
                    with col2:
                        if debug_info.get('prompt_improvements'):
                            st.write("**Prompt Improvements:**")
                            st.success(debug_info['prompt_improvements'])
    
    def display_judgements(self, archive_id: str):
        """Display judgements for an archive entry"""
        judgements = self.judgements_by_archive.get(archive_id, [])
        
        if not judgements:
            st.info("No judgements found for this archive entry.")
            return
        
        st.subheader(f"⚖️ Judgements ({len(judgements)})")
        
        for i, judgement in enumerate(judgements):
            with st.expander(f"Judgement {i+1}: Score {judgement.get('judge_score', 'N/A')}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Model New:**", judgement.get('model_new', ''))
                    st.write("**Model Original:**", judgement.get('model_original', ''))
                    st.write("**Judge Score:**", judgement.get('judge_score', ''))
                    st.write("**DateTime:**", self.format_timestamp(judgement.get('datetime', '')))
                
                with col2:
                    st.write("**Original Response:**")
                    original_response = judgement.get('original_response', {})
                    if isinstance(original_response, dict):
                        st.json(original_response)
                    else:
                        st.code(str(original_response), language='json')
                    
                    st.write("**New Response:**")
                    new_response = judgement.get('new_response', {})
                    if isinstance(new_response, dict):
                        st.json(new_response)
                    else:
                        st.code(str(new_response), language='json')
                
                # Judge reasoning and improvements
                st.write("**Judge Reasoning:**")
                st.info(judgement.get('judge_reasoning', ''))
                
                improvements = judgement.get('judge_improvements')
                if improvements:
                    st.write("**Judge Improvements:**")
                    st.success(improvements)
    
    def run(self):
        """Main application logic"""
        st.markdown('<h1 class="main-header">📊 Archive Viewer</h1>', unsafe_allow_html=True)
        
        # Sidebar for controls
        with st.sidebar:
            st.header("🔍 Search & Filter")
            
            # Search
            search_term = st.text_input("Search in all fields:", placeholder="Enter search term...")
            
            # Model filter
            models = list(set([item.get('model', '') for item in self.archived_calls if item.get('model')]))
            models.sort()
            selected_model = st.selectbox("Filter by Model:", ['All'] + models)
            
            # Tag filter
            tags = list(set([item.get('tag', '') for item in self.archived_calls if item.get('tag')]))
            tags.sort()
            selected_tag = st.selectbox("Filter by Tag:", ['All'] + tags)
            
            # Pagination settings
            st.header("📄 Pagination")
            items_per_page = st.selectbox("Items per page:", [10, 25, 50, 100], index=1)
            
            # Statistics
            st.header("📈 Statistics")
            st.metric("Total Archives", len(self.archived_calls))
            st.metric("Total Eval Runs", len(self.eval_runs))
            st.metric("Total Judgements", len(self.judgement_results))
        
        # Filter data
        filtered_data = self.archived_calls
        
        # Apply search
        if search_term:
            filtered_data = self.search_data(filtered_data, search_term)
        
        # Apply model filter
        if selected_model != 'All':
            filtered_data = [item for item in filtered_data if item.get('model') == selected_model]
        
        # Apply tag filter
        if selected_tag != 'All':
            filtered_data = [item for item in filtered_data if item.get('tag') == selected_tag]
        
        # Create DataFrame
        df = self.create_archive_dataframe(filtered_data)
        
        if df.empty:
            st.warning("No data found matching the current filters.")
            return
        
        # Pagination
        total_items = len(df)
        total_pages = (total_items + items_per_page - 1) // items_per_page
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            page = st.selectbox(f"Page (1-{total_pages}):", range(1, total_pages + 1), index=0)
        
        start_idx = (page - 1) * items_per_page
        end_idx = start_idx + items_per_page
        page_data = df.iloc[start_idx:end_idx]
        
        # Display results info
        st.info(f"Showing {len(page_data)} of {total_items} results (Page {page} of {total_pages})")
        
        # Main table
        st.subheader("📋 Archive Entries")
        
        # Create a copy of the dataframe for display (without raw data)
        display_df = page_data.drop('Raw Data', axis=1)
        
        # Display the table
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )
        
        # Detailed view
        st.subheader("🔍 Detailed View")
        
        if not page_data.empty:
            selected_index = st.selectbox(
                "Select an entry to view details:",
                range(len(page_data)),
                format_func=lambda x: f"{page_data.iloc[x]['ID']} - {page_data.iloc[x]['Model']}"
            )
            
            selected_data = page_data.iloc[selected_index]['Raw Data']
            archive_id = selected_data.get('id')
            
            # Display archive details
            self.display_archive_details(selected_data)
            
            # Display eval runs
            self.display_eval_runs(archive_id)
            
            # Display judgements
            self.display_judgements(archive_id)

def main():
    """Main function"""
    try:
        viewer = ArchiveViewer()
        viewer.run()
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.exception(e)

if __name__ == "__main__":
    main() 
