import streamlit as st
import tempfile
import os
import json
import helper  # Uncomment when you have the helper module

def format_analysis_display(analysis_result):
    """
    Display the analysis results using Streamlit components.
    
    Args:
        analysis_result (dict): The analysis result dictionary
    """
    # Display total duration
    st.markdown("---")
    st.markdown("### 📊 Video Analysis Results")
    st.markdown("---")
    
    st.metric("Total Duration", f"{analysis_result['total_duration']} seconds")
    
    # Display activity segments
    st.markdown("---")
    st.markdown("### 🎬 Activity Segments")
    st.markdown("---")
    
    for i, segment in enumerate(analysis_result['activity_segments'], 1):
        with st.container():
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"**Segment {i}**")
                st.markdown(f"⏱️ **Time:** {segment['start_time']}s - {segment['end_time']}s")
                st.markdown(f"🎯 **Activity:** {segment['activity_description']}")
            
            with col2:
                st.markdown(f"**Confidence**")
                # Handle confidence as integer (not percentage string)
                confidence_value = segment['confidence']
                st.progress(confidence_value / 100)
                st.markdown(f"*{confidence_value}%*")
            
            st.markdown("---")

def main():
    st.set_page_config(
        page_title="Video Activity Analysis",
        page_icon="🎥",
        layout="wide"
    )
    
    st.title("🎥 Video Activity Analysis")
    st.markdown("Upload an MP4 video to analyze human activities")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a video file",
        type=['mp4'],
        help="Upload an MP4 video file for activity analysis"
    )
    
    if uploaded_file is not None:
        # Display video preview
        st.video(uploaded_file)
        
        # Predict button
        if st.button("🔍 Predict Activities", type="primary"):
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(uploaded_file.read())
                video_file_path = tmp_file.name
            
            try:
                # Show spinner while processing
                with st.spinner("Analyzing video... This may take a moment."):
                    # Call the helper function
                    json_response = helper.get_human_activity(video_file_path)
                    
                    # Demo response - Remove this when using actual helper
                    # json_response = {
                    #     'total_duration': 18,
                    #     'activity_segments': [
                    #         {
                    #             'start_time': 0,
                    #             'end_time': 5,
                    #             'activity_description': 'person is cooking at stove',
                    #             'confidence': 95
                    #         },
                    #         {
                    #             'start_time': 5,
                    #             'end_time': 10,
                    #             'activity_description': 'person is standing near window',
                    #             'confidence': 90
                    #         },
                    #         {
                    #             'start_time': 10,
                    #             'end_time': 14,
                    #             'activity_description': 'person is walking towards stove',
                    #             'confidence': 92
                    #         },
                    #         {
                    #             'start_time': 14,
                    #             'end_time': 18,
                    #             'activity_description': 'person is looking at stove',
                    #             'confidence': 93
                    #         }
                    #     ]
                    # }
                
                # Display success message
                st.success("✅ Analysis completed successfully!")
                
                # Display formatted results
                format_analysis_display(json_response)
                
                # Show raw JSON in expander
                with st.expander("📄 View Raw JSON Response"):
                    st.json(json_response)
                
            except Exception as e:
                st.error(f"❌ An error occurred during analysis: {str(e)}")
            
            finally:
                # Clean up temporary file
                if os.path.exists(video_file_path):
                    os.unlink(video_file_path)
    else:
        st.info("👆 Please upload a video file to begin")

if __name__ == "__main__":
    main()
