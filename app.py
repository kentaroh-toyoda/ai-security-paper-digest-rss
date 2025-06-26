#!/usr/bin/env python3
"""
Streamlit web app for CRUD operations on the Qdrant KB.
"""

import streamlit as st
import pandas as pd
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from utils.qdrant import (
    init_qdrant_client,
    ensure_collection_exists,
    get_all_papers,
    generate_point_id,
    COLLECTION_NAME
)
from qdrant_client.http.models import Filter, FieldCondition, MatchValue

# Load environment variables
load_dotenv()

# Set page config
st.set_page_config(
    page_title="Paper Digest",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Qdrant client
@st.cache_resource
def get_qdrant_client():
    client = init_qdrant_client()
    # ensure_collection_exists(client)
    return client

# Get all papers
@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_papers():
    client = get_qdrant_client()
    papers = get_all_papers(client)
    # Sort papers by date (newest to oldest)
    papers = sorted(papers, key=lambda x: x.get('date', ''), reverse=True)
    return papers

def filter_papers(papers: List[Dict[str, Any]],
                  search_text: Optional[str] = None,
                  authors: Optional[str] = None,
                  tags: Optional[str] = None,
                  modalities: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Filter papers based on search criteria.
    """
    filtered_papers = papers

    # Filter by search text (in title or abstract)
    if search_text:
        search_terms = search_text.lower().split()
        filtered_papers = [
            paper for paper in filtered_papers
            if any(
                term in paper.get('title', '').lower() or
                term in paper.get('abstract', '').lower()
                for term in search_terms
            )
        ]

    # Filter by authors
    if authors:
        author_list = [a.strip().lower() for a in authors.split(',')]
        filtered_papers = [
            paper for paper in filtered_papers
            if any(author.lower() in paper.get('authors', '').lower() for author in author_list)
        ]

    # Filter by tags
    if tags:
        tag_list = [t.strip().lower() for t in tags.split(',')]
        
        # Split into positive and negative tags
        positive_tags = [tag for tag in tag_list if not tag.startswith('!')]
        negative_tags = [tag[1:] for tag in tag_list if tag.startswith('!')]
        
        # Filter for positive tags (papers must have at least one of these tags)
        if positive_tags:
            filtered_papers = [
                paper for paper in filtered_papers
                if any(tag.lower() in [t.lower() for t in paper.get('tags', [])] for tag in positive_tags)
            ]
        
        # Filter out negative tags (papers must not have any of these tags)
        if negative_tags:
            filtered_papers = [
                paper for paper in filtered_papers
                if not any(neg_tag.lower() in [t.lower() for t in paper.get('tags', [])] for neg_tag in negative_tags)
            ]

    # Filter by modalities
    if modalities:
        modality_list = [m.strip().lower() for m in modalities.split(',')]
        filtered_papers = [
            paper for paper in filtered_papers
            if any(modality.lower() in [m.lower() for m in paper.get('modalities', [])] for modality in modality_list)
        ]

    return filtered_papers

def add_tag_to_paper(client, paper_url: str, new_tag: str) -> bool:
    """Add a tag to a paper in Qdrant."""
    try:
        # Find the point by URL
        response = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="url",
                        match=MatchValue(value=paper_url)
                    )
                ]
            ),
            limit=1
        )
        
        if not response[0]:
            return False
        
        point = response[0][0]
        
        # Get existing tags or initialize empty list
        existing_tags = point.payload.get("tags", [])
        
        # Check if tag already exists
        if new_tag in existing_tags:
            return True
        
        # Add the new tag
        all_tags = existing_tags + [new_tag]
        
        # Update the point with the new tags
        client.set_payload(
            collection_name=COLLECTION_NAME,
            payload={"tags": all_tags},
            points=[point.id]
        )
        
        return True
    except Exception as e:
        st.error(f"Error adding tag: {str(e)}")
        return False

def delete_tag_from_paper(client, paper_url: str, tag_to_delete: str) -> bool:
    """Delete a tag from a paper in Qdrant."""
    try:
        # Find the point by URL
        response = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="url",
                        match=MatchValue(value=paper_url)
                    )
                ]
            ),
            limit=1
        )
        
        if not response[0]:
            return False
        
        point = response[0][0]
        
        # Get existing tags or initialize empty list
        existing_tags = point.payload.get("tags", [])
        
        # Check if the tag exists
        if tag_to_delete not in existing_tags:
            return False
        
        # Remove the tag
        updated_tags = [tag for tag in existing_tags if tag != tag_to_delete]
        
        # Update the point with the new tags
        client.set_payload(
            collection_name=COLLECTION_NAME,
            payload={"tags": updated_tags},
            points=[point.id]
        )
        
        return True
    except Exception as e:
        st.error(f"Error deleting tag: {str(e)}")
        return False

def update_paper_field(client, paper_url: str, field: str, new_value: Any) -> bool:
    """Update a field in a paper in Qdrant."""
    try:
        # Find the point by URL
        response = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="url",
                        match=MatchValue(value=paper_url)
                    )
                ]
            ),
            limit=1
        )
        
        if not response[0]:
            return False
        
        point = response[0][0]
        
        # Update the field
        client.set_payload(
            collection_name=COLLECTION_NAME,
            payload={field: new_value},
            points=[point.id]
        )
        
        return True
    except Exception as e:
        st.error(f"Error updating {field}: {str(e)}")
        return False

def create_dataframe_from_papers(papers: List[Dict[str, Any]]) -> pd.DataFrame:
    """Create a DataFrame from papers for table display."""
    # Prepare data for DataFrame
    data = []
    for paper in papers:
        # Get basic paper info
        title = paper.get('title', 'N/A')
        date = paper.get('date', 'N/A')
        # Truncate the time part from the date
        if isinstance(date, str) and 'T' in date:
            date = date.split('T')[0]
        url = paper.get('url', 'N/A')
        
        # Get modalities and tags as strings
        modalities = ", ".join(paper.get('modalities', []))
        tags = ", ".join(paper.get('tags', []))
        
        # Add to data
        data.append({
            'Date': date,
            'Title': title,
            'URL': url,
            'Modalities': modalities,
            'Tags': tags
        })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    return df

def main():
    st.title("📚 Paper Digest")
    
    # Initialize session state for selected paper
    if "selected_paper_url" not in st.session_state:
        st.session_state.selected_paper_url = None
    
    # Sidebar for filters
    st.sidebar.header("Search Filters")
    
    # Search text
    search_text = st.sidebar.text_input("Search in Title/Abstract")
    
    # Author filter
    authors_filter = st.sidebar.text_input("Filter by Authors (comma-separated)")
    
    # Tags filter
    tags_filter = st.sidebar.text_input("Filter by Tags (comma-separated, use ! for exclusion)")
    
    # Modalities filter
    modalities_filter = st.sidebar.text_input("Filter by Modalities (comma-separated)")
    
    # Hidden display options (set defaults)
    view_mode = "Table"
    show_abstract = False
    show_summary = False
    
    # Load papers
    with st.spinner("Loading papers..."):
        all_papers = load_papers()
    
    # Filter papers
    filtered_papers = filter_papers(
        all_papers,
        search_text=search_text,
        authors=authors_filter,
        tags=tags_filter,
        modalities=modalities_filter
    )
    
    # Display paper count
    st.write(f"Showing {len(filtered_papers)} of {len(all_papers)} papers")
    
    # Create a container for the papers
    papers_container = st.container()
    
    # Display papers based on view mode
    with papers_container:
        if not filtered_papers:
            st.info("No papers found matching the search criteria.")
        else:
            if view_mode == "Table":
                # Create DataFrame for table view
                df = create_dataframe_from_papers(filtered_papers)
                
                # Display table
                st.dataframe(
                    df,
                    column_config={
                        "URL": st.column_config.LinkColumn(),
                        "Title": st.column_config.TextColumn(width="large"),
                        "Date": st.column_config.TextColumn(width="small"),
                        "Modalities": st.column_config.TextColumn(width="medium"),
                        "Tags": st.column_config.TextColumn(width="medium")
                    },
                    use_container_width=True,
                    hide_index=True
                )
                
                # Paper details section
                st.subheader("Paper Details")
                
                # Use a selectbox to select a paper
                selected_paper_url = st.selectbox(
                    "Select a paper to view/edit details",
                    options=[paper.get('url', '') for paper in filtered_papers],
                    format_func=lambda x: next((paper.get('title', 'Unknown') for paper in filtered_papers if paper.get('url', '') == x), 'Unknown')
                )
                
                # Store the selected paper URL in session state
                st.session_state.selected_paper_url = selected_paper_url
                
                if selected_paper_url:
                    # Find the selected paper
                    selected_paper = next((paper for paper in filtered_papers if paper.get('url', '') == selected_paper_url), None)
                    
                    if selected_paper:
                        # Display paper details in tabs
                        tabs = st.tabs(["Info", "Tags", "Abstract/Summary"])
                        
                        # Info tab
                        with tabs[0]:
                            st.write("**Title:**", selected_paper.get('title', 'N/A'))
                            st.write("**Date:**", selected_paper.get('date', 'N/A').split('T')[0] if isinstance(selected_paper.get('date', 'N/A'), str) and 'T' in selected_paper.get('date', 'N/A') else selected_paper.get('date', 'N/A'))
                            st.write("**URL:**")
                            st.markdown(f"[{selected_paper_url}]({selected_paper_url})")
                            st.write("**Authors:**", selected_paper.get('authors', 'N/A'))
                            st.write("**Modalities:**", ", ".join(selected_paper.get('modalities', [])) or "N/A")
                            
                            # Edit modalities
                            st.subheader("Edit Modalities")
                            modalities_str = ", ".join(selected_paper.get('modalities', []))
                            new_modalities = st.text_input("Modalities (comma-separated)", value=modalities_str)
                            if st.button("Update Modalities"):
                                # Convert string to list
                                modalities_list = [m.strip() for m in new_modalities.split(',') if m.strip()]
                                client = get_qdrant_client()
                                if update_paper_field(client, selected_paper_url, "modalities", modalities_list):
                                    st.success("Modalities updated")
                                    st.cache_data.clear()
                                    st.rerun()
                                else:
                                    st.error("Failed to update modalities")
                        
                        # Tags tab
                        with tabs[1]:
                            st.subheader("Current Tags")
                            tags_list = selected_paper.get('tags', [])
                            if tags_list:
                                for tag in tags_list:
                                    col1, col2 = st.columns([3, 1])
                                    with col1:
                                        st.write(f"• {tag}")
                                    with col2:
                                        if st.button("Delete", key=f"delete_tag_{tag}"):
                                            client = get_qdrant_client()
                                            if delete_tag_from_paper(client, selected_paper_url, tag):
                                                st.success(f"Tag '{tag}' deleted")
                                                st.cache_data.clear()
                                                st.rerun()
                                            else:
                                                st.error(f"Failed to delete tag '{tag}'")
                            else:
                                st.write("No tags")
                            
                            # Add new tags
                            st.subheader("Add New Tags")
                            new_tags = st.text_input("New tags (comma-separated)")
                            if st.button("Add Tags"):
                                if new_tags:
                                    # Split by comma and strip whitespace
                                    tag_list = [tag.strip() for tag in new_tags.split(',') if tag.strip()]
                                    
                                    if not tag_list:
                                        st.warning("Please enter at least one tag")
                                    else:
                                        client = get_qdrant_client()
                                        success_count = 0
                                        
                                        # Add each tag individually
                                        for tag in tag_list:
                                            if add_tag_to_paper(client, selected_paper_url, tag):
                                                success_count += 1
                                        
                                        if success_count > 0:
                                            if success_count == len(tag_list):
                                                st.success(f"All {success_count} tags added successfully")
                                            else:
                                                st.warning(f"{success_count} out of {len(tag_list)} tags added successfully")
                                            st.cache_data.clear()
                                            st.rerun()
                                        else:
                                            st.error("Failed to add any tags")
                                else:
                                    st.warning("Please enter at least one tag")
                        
                        # Abstract/Summary tab
                        with tabs[2]:
                            # Abstract
                            st.subheader("Abstract")
                            abstract = selected_paper.get('abstract', 'N/A')
                            st.write(abstract)
                            
                            # Summary
                            st.subheader("Summary")
                            summary = selected_paper.get('summary', [])
                            if summary:
                                for point in summary:
                                    st.write(f"• {point}")
                            else:
                                st.write("No summary available")
                            
                            # Edit summary
                            st.subheader("Edit Summary")
                            summary_str = "\n".join(selected_paper.get('summary', []))
                            new_summary = st.text_area("Summary (one point per line)", value=summary_str, height=200)
                            if st.button("Update Summary"):
                                # Convert string to list
                                summary_list = [line.strip() for line in new_summary.split('\n') if line.strip()]
                                client = get_qdrant_client()
                                if update_paper_field(client, selected_paper_url, "summary", summary_list):
                                    st.success("Summary updated")
                                    st.cache_data.clear()
                                    st.rerun()
                                else:
                                    st.error("Failed to update summary")
            
            else:  # Detailed view
                # Create an expander for each paper
                for i, paper in enumerate(filtered_papers):
                    # Get basic paper info
                    title = paper.get('title', 'N/A')
                    date = paper.get('date', 'N/A')
                    # Truncate the time part from the date
                    if isinstance(date, str) and 'T' in date:
                        date = date.split('T')[0]
                    url = paper.get('url', 'N/A')
                    authors = paper.get('authors', 'N/A')
                    
                    # Get modalities and tags as lists
                    modalities_list = paper.get('modalities', [])
                    tags_list = paper.get('tags', [])
                    
                    # Create expander with title and date
                    with st.expander(f"{title} ({date})"):
                        # Two columns for metadata
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**URL:**")
                            st.markdown(f"[{url}]({url})")
                            
                            st.write("**Authors:**")
                            st.write(authors)
                            
                            st.write("**Modalities:**")
                            if modalities_list:
                                st.write(", ".join(modalities_list))
                            else:
                                st.write("N/A")
                            
                            # Edit modalities
                            modalities_str = ", ".join(modalities_list)
                            new_modalities = st.text_input("Edit modalities (comma-separated)", value=modalities_str, key=f"modalities_{i}")
                            if st.button("Update Modalities", key=f"update_modalities_{i}"):
                                # Convert string to list
                                modalities_list = [m.strip() for m in new_modalities.split(',') if m.strip()]
                                client = get_qdrant_client()
                                if update_paper_field(client, url, "modalities", modalities_list):
                                    st.success("Modalities updated")
                                    st.cache_data.clear()
                                    st.rerun()
                                else:
                                    st.error("Failed to update modalities")
                        
                        with col2:
                            st.write("**Tags:**")
                            
                            # Display current tags
                            if tags_list:
                                # Create a container for tags
                                tags_container = st.container()
                                with tags_container:
                                    for tag in tags_list:
                                        col_tag, col_delete = st.columns([3, 1])
                                        with col_tag:
                                            st.write(f"• {tag}")
                                        with col_delete:
                                            if st.button("Delete", key=f"delete_{i}_{tag}"):
                                                client = get_qdrant_client()
                                                if delete_tag_from_paper(client, url, tag):
                                                    st.success(f"Tag '{tag}' deleted")
                                                    st.cache_data.clear()
                                                    st.rerun()
                                                else:
                                                    st.error(f"Failed to delete tag '{tag}'")
                            else:
                                st.write("No tags")
                            
                            # Add new tags
                            new_tags = st.text_input("Add new tags (comma-separated)", key=f"new_tags_{i}")
                            if st.button("Add Tags", key=f"add_tags_{i}"):
                                if new_tags:
                                    # Split by comma and strip whitespace
                                    tag_list = [tag.strip() for tag in new_tags.split(',') if tag.strip()]
                                    
                                    if not tag_list:
                                        st.warning("Please enter at least one tag")
                                    else:
                                        client = get_qdrant_client()
                                        success_count = 0
                                        
                                        # Add each tag individually
                                        for tag in tag_list:
                                            if add_tag_to_paper(client, url, tag):
                                                success_count += 1
                                        
                                        if success_count > 0:
                                            if success_count == len(tag_list):
                                                st.success(f"All {success_count} tags added successfully")
                                            else:
                                                st.warning(f"{success_count} out of {len(tag_list)} tags added successfully")
                                            st.cache_data.clear()
                                            st.rerun()
                                        else:
                                            st.error("Failed to add any tags")
                                else:
                                    st.warning("Please enter at least one tag")
                        
                        # Show abstract if selected
                        if show_abstract:
                            st.write("**Abstract:**")
                            st.write(paper.get('abstract', 'N/A'))
                        
                        # Show summary if selected
                        if show_summary:
                            st.write("**Summary:**")
                            summary = paper.get('summary', [])
                            if summary:
                                for point in summary:
                                    st.write(f"• {point}")
                            else:
                                st.write("No summary available")
                            
                            # Edit summary
                            summary_str = "\n".join(summary)
                            new_summary = st.text_area("Edit summary (one point per line)", value=summary_str, key=f"summary_{i}", height=150)
                            if st.button("Update Summary", key=f"update_summary_{i}"):
                                # Convert string to list
                                summary_list = [line.strip() for line in new_summary.split('\n') if line.strip()]
                                client = get_qdrant_client()
                                if update_paper_field(client, url, "summary", summary_list):
                                    st.success("Summary updated")
                                    st.cache_data.clear()
                                    st.rerun()
                                else:
                                    st.error("Failed to update summary")

if __name__ == "__main__":
    main()
