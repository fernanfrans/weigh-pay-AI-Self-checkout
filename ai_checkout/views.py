from django.shortcuts import render
from django.http import StreamingHttpResponse # import StreamingHttpResponse for video feed
import detection
import pandas as pd
from django.http import JsonResponse


# landing page of the website
def landingpage(request):
    return render(request, 'landingpage.html')

# Main page of the website
def main(request):
    return render(request, 'main.html')

# function to stream the video feed
def video_feed(request):
    return StreamingHttpResponse(detection.main(), content_type='multipart/x-mixed-replace; boundary=frame', status=200)

#function for table
def result_table(request):
    # Get the detection results
    detection_results = detection.get_detection_results()
    
    if not detection_results:
        table_html = "<p>No detection results available.</p>"
    else:
        # Create DataFrame from the list of dictionaries
        df = pd.DataFrame(detection_results)
        table_html = df.to_html(classes='table table-striped', index=False)

    return JsonResponse({'table_html': table_html})