import csv
import xml.etree.ElementTree as ET
import pandas as pd
import argparse





def extract_bbox_filtered_write_csv(xml_file_gt,csv_file_gt):
    # Parse the XML file
    file = open(csv_file_gt, 'w') # Open the output CSV file for writing
    class_list = [] # Create an empty list to store class labels
    tree = ET.parse(xml_file_gt) # Parse the XML file using ElementTree library
    root = tree.getroot() # Get the root element of the XML file
    frames = root.findall('frame') # Find all the 'frame' elements in the XML file
    ignored_region = root.find('ignored_region') # Find the 'ignored_region' element in the XML file
    ignored_boxes = ignored_region.findall('box') # Find all the 'box' elements inside 'ignored_region'

    # Loop through each frame in the XML file
    for frame in frames:
        target_list = frame.find('target_list') # Find the 'target_list' element in the current frame   
        for target in target_list:
            box = target.find('box') # Find the 'box' element inside 'target'
            ignored=False
            for ignored_box in ignored_boxes:
                # Get the top-left coordinates (xr,yr) and width/height (wr,hr) of the ignored box
                xr , yr ,wr ,hr = map(float,(ignored_box.attrib['left'],ignored_box.attrib['top'],ignored_box.attrib['width'],ignored_box.attrib['height']) ) 
                # Get the top-left coordinates (x,y) and width/height (w,h) of the bounding box
                x , y ,w ,h = map(float,(box.attrib['left'],box.attrib['top'],box.attrib['width'],box.attrib['height']) )
                x2, y2 = x + w, y + h 
                # Check if the bounding box falls within the specified region
                ignored = ignored or (x >= xr and y >= yr and x2 <= xr + wr and y2 <= yr+hr)

            if not ignored:
                file.write(frame.attrib['num'] + ',' + target.attrib['id'] + ',' + box.attrib['left'] + ',' + box.attrib['top'] + ',' + box.attrib['width'] + ',' + box.attrib['height'] + ',1,-1,-1,-1\n')

    # Close the output CSV file
    file.close()



def filter_region_model_output(xml_file_gt,model_output,model_output_filtered):
    # Define the top left coordinates and width/height of the region to remove bounding boxes from
    tree = ET.parse(xml_file_gt) # Parse the XML file using ElementTree library
    root = tree.getroot() # Get the root element of the XML file
    ignored_region = root.find('ignored_region') # Find the 'ignored_region' element in the XML file
    ignored_boxes = ignored_region.findall('box') # Find all the 'box' elements inside 'ignored_region'



    # set the file path and load the csv data into a pandas dataframe
    df = pd.read_csv(model_output,header=None)

    # remove rows that meet a specific condition
    for ignored_box in ignored_boxes:
                # Get the top-left coordinates (xr,yr) and width/height (wr,hr) of the ignored box
                xr , yr ,wr ,hr = map(float,(ignored_box.attrib['left'],ignored_box.attrib['top'],ignored_box.attrib['width'],ignored_box.attrib['height']) ) 
                df = df.loc[~((df[2] >= xr) & (df[3] >= yr) &
                (df[2] + df[4] <= xr + wr) &
                (df[3] + df[5] <= yr+hr))]




    # write the updated dataframe back to the original file
    df.to_csv(model_output_filtered, header=False,index=False)


def main(xml_file_gt,csv_file_gt,model_output,model_output_filtered):
    extract_bbox_filtered_write_csv(xml_file_gt,csv_file_gt)
    filter_region_model_output(xml_file_gt,model_output,model_output_filtered)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Prepare.")
    parser.add_argument("--gt_xml",
                        help="ground truth annotation in xml format ",
                        default="MVI_39031.xml")
    parser.add_argument("--gt_csv",
                        help="ground truth annotation ignored region filterd in csv format",
                        default="MVI_39031.csv")
    parser.add_argument("--model_out",
                        help="model output annotation",
                        default="output.csv")
    parser.add_argument("--model_out_filtered",
                        help="model output annotation with gt ingnored region filtered",
                        default="output_filtered.csv")
    args = parser.parse_args()
    xml_file_gt=args.gt_xml
    csv_file_gt=args.gt_csv
    model_output=args.model_out
    model_output_filtered=args.model_out_filtered


    main(xml_file_gt,csv_file_gt,model_output,model_output_filtered)