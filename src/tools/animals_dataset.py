# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 18:40:49 2020

@author: Frank
"""
import torch
import numpy as np
from PIL import Image, ImageDraw

import torchvision.transforms as T
import pandas as pd   # for csv reading into dataframes
import os             # for path building and reading of images

import sys
import cv2 as cv
from torchvision.transforms import functional as F


 #########################
## Load based on videos, but load only one image at a time

# creating own dataset, reading images in COCO style and doing transformations for learning process
class AnimalDatasetImagecsv(object):
    def __init__(self, images, annotations, datasettype,splitindices=None,preprocessed=False,intermediatedata=False):
        self.imgpath = images
        self.annpath =annotations
        self.intermediatedata=intermediatedata
        if(preprocessed):
            self.csvdata = pd.read_csv(self.annpath)   
        else:

            # load the cocodata File for the annotations of each video
            self.csvdata = pd.read_csv(self.annpath)
            # rename the column, because it represents the video_number
            self.csvdata.rename(columns={'file_attributes':'video_number'}, inplace=True)
            self.csvdata.rename(columns={'region_attributes':'class'}, inplace=True)
            self.csvdata.rename(columns={'region_shape_attributes':'xpoints'}, inplace=True)
    
            if(datasettype=="Animals"):
                self.classdict= {"deer":1, "boar":2, "fox":3, "hare":4}
            elif(datasettype=="BayWald"):
                self.classdict= {"roe deer":1, "red deer":2}
            elif(datasettype=="Wildpark"):
                self.classdict= {"red deer":1, "fallow deer":2} 
            # for splitting attribut values, insert the new columns
            self.csvdata['track'] = pd.Series(np.random.randn(self.csvdata.shape[0]), index=self.csvdata.index)
            ### if you deal with intermediate data you also have a score value saved
            if(intermediatedata):
                self.csvdata['score'] = pd.Series(np.random.randn(self.csvdata.shape[0]), index=self.csvdata.index)
            
            self.csvdata.insert(6,"ypoints",pd.Series(np.random.randn(self.csvdata.shape[0]), index=self.csvdata.index))
            # for giving each image file a unique image_id
            self.csvdata.insert(3,"image_id",pd.Series(np.random.randn(self.csvdata.shape[0]), index=self.csvdata.index))
            # groupby the filenames for generating a dictionary of corresponding image ids
            filegroup= self.csvdata.groupby(self.csvdata["filename"])
            num=np.arange(filegroup.ngroups)  
            imgid_dict= dict(zip(filegroup.groups.keys(),num))
            
    
    
    
            # preprocess the data from the csv for better reading from the dataframe
            for i in range(self.csvdata.shape[0]):
                # the if case is just interesting for datasets where files are that do not contain annotations
                if(int(self.csvdata.loc[i,"region_count"])>0):
                    # write just the Video number int in the row for better accessing of the values
                    p=self.csvdata.loc[i, "video_number"]
                    val=[int(s) for s in p.split("\"") if s.isdigit()]
                    self.csvdata.loc[i, "video_number"]=val[0]            
                    s=self.csvdata.loc[i,"xpoints"]
                    sp= s.split("[")
        
                    x_points= sp[1].split("]")[0]
                    y_points= sp[2].split("]")[0]
                    # concatenate the x and y points by just a ; for easier extraction later on 
                
                    self.csvdata.loc[i,"xpoints"]=x_points
                    self.csvdata.loc[i,"ypoints"]=y_points
                    
                    #prepare the region attributes column for better usage
                    r=self.csvdata.loc[i,"class"]
                    rs=r.split("\"")
        
                    self.csvdata.loc[i,"class"]= self.classdict[rs[3]]
                    self.csvdata.loc[i,"track"]=int(rs[7])
                    ### for intermediate data also read the 
                    if(intermediatedata):
                        self.csvdata.loc[i,"score"]=float(rs[11])    
                    #print(self.csvdata.loc[i,"region_attributes"])
                    
                    # insert image ids
                    self.csvdata.loc[i,"image_id"] =int(imgid_dict[self.csvdata.loc[i,"filename"]])
    
            # filter out the rows where are no annotations
            self.csvdata = self.csvdata[self.csvdata["region_count"] !=0]        
    #        vidgrouped= self.csvdata.groupby(self.csvdata["filename"])
    #        for name, g in vidgrouped:
    #
    #            if(len(g)<1):
    #                print(name)
    #                print(len(g))
    
        
        if(splitindices is not None):
            
            self.csvdata=self.csvdata[self.csvdata["video_number"].isin(splitindices)]
            self.len = len(splitindices)
            videoids=list(np.arange(len(splitindices)))
            self.videoid_dict=dict(zip(videoids,splitindices))

        else:
         
            # get number of different video ids
            vidgroup= self.csvdata.groupby(self.csvdata["video_number"])
            self.len = vidgroup.ngroups
            videoids=list(np.arange(self.len))
            # create a dict with the unique video numbers combined with the indices starting with 0
            self.videoid_dict=dict(zip(videoids,self.csvdata["video_number"].unique()))


        
        
        # get the image by filename from csv
    def getframe(self, filename_input):
        # extract the corresponding frames 
        vidlist= self.csvdata.loc[self.csvdata['filename'] == filename_input]
   
  
        imfile=os.path.join(self.imgpath,filename_input)

        img= Image.open(imfile)

        # Get the number of objects / animals by extracting the region count value
        #num_objs= vidlist.iloc[0].loc["region_count"]
        num_objs=vidlist.shape[0]
#        print("hhh")
#        print(num_objs)
#        print(vidlist.shape[0])
   
        boxes=[]        
        # generate the binary masks
        masks=np.zeros((num_objs,img.size[1],img.size[0]),dtype=np.uint8)
        # area of the segments and iscrowd attribute
        area=torch.zeros((num_objs,),dtype=torch.float32)
        iscrowd =torch.zeros((num_objs,), dtype=torch.int64)
        # save the labels
        labels=torch.zeros((num_objs,), dtype=torch.int64)
        
        # save the track number
        tracks=torch.zeros((num_objs,), dtype=torch.int64)
        
        if(self.intermediatedata):
            scores=torch.zeros((num_objs,))
        
        # count the segments
        count=0
  
        for _, frame in vidlist.iterrows():
           
            #print(frame)
            # extract the polygon points and split by defined marker ;
 
            xpoint_str=frame.loc["xpoints"]
            ypoint_str=frame.loc["ypoints"]
            
            # convert to int list
            xpoints=list(map(int, xpoint_str.split(',')))
            ypoints=list(map(int, ypoint_str.split(',')))
            
            # generate the mask from the polyline
            points=[]
            for j in range(len(xpoints)):
                points.append([xpoints[j],ypoints[j]])
            points=np.asarray(points)
      
            imgMask = np.zeros( (img.size[1],img.size[0]) ,dtype=np.uint8) # create a single channel 200x200 pixel black image 
            cv.fillPoly(imgMask, pts =[points], color=(1,1,1))
            masks[count] = np.array(imgMask)
            # get the area of the segment
            area[count]=cv.countNonZero(masks[count])
            # is crowd always to 0, should indicate overlap, but is here not interesting for us
            iscrowd[count]=0
            
            
            # extract the bounding box information from the polyline
            xmin = min(xpoints)
            ymin = min(ypoints)
            xmax = max(xpoints)
            ymax = max(ypoints)
            boxes.append([xmin,ymin,xmax,ymax])
            
            # set the label
            labels[count]=frame.loc["class"]
            # set the track number
            tracks[count]=frame.loc["track"]
            if(self.intermediatedata):
                scores[count]=frame.loc["score"]
            
            count+=1

        # convert the np array to a tensor
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        # convet the bounding boxes to tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
 

        # generate image id part, not really relevant    
        #filename=  frame.loc["filename"] 
        image_id=  frame.loc["image_id"] 
        
            
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        if(self.intermediatedata):
            masks=masks.view(masks.shape[0],1,masks.shape[1],masks.shape[2])
        target["masks"] = masks
        target["image_id"] = torch.tensor([image_id]) 
        #target["filename"] = filename
        target["area"] = area
        target["iscrowd"] = iscrowd
        target["track"]=tracks
        if(self.intermediatedata):
            target["scores"]=scores    
      
        # convert image back to RGB, because the reid model and other models need it in this way
        img=img.convert("RGB")
        # in transforms the PIL image will be converted to a pytorch tensor
        img=F.to_tensor(img)


            

        return img, target
    # get a list with the corresponding ids of the frames from video vid_idx
    def __getitem__(self, vid_idx):
        # if a train test split was done videoid 1 might not refer to video 1 anymore, so check the videodictid
        vid_idx2=self.videoid_dict[vid_idx]
        vidlist= self.csvdata.loc[self.csvdata['video_number'] == vid_idx2]
        #print(vidlist)
        # group by the image names, because there might be multiple rows when there is more than one object annotated in the video
        vidgrouped= vidlist.groupby(vidlist["filename"])
   
        
        return list(vidgrouped.groups.keys())

    def __len__(self):
        return self.len   
    
# preprocess the csv data and save it in "right" format 
def preprocess_data(path,annotationfile, datasettype):
        annotations=os.path.join(path,annotationfile)
        # load the cocodata File for the annotations of each video
        csvdata = pd.read_csv(annotations)
        # rename the column, because it represents the video_number
        csvdata.rename(columns={'file_attributes':'video_number'}, inplace=True)
        csvdata.rename(columns={'region_attributes':'class'}, inplace=True)
        csvdata.rename(columns={'region_shape_attributes':'xpoints'}, inplace=True)

        if(datasettype=="Animals"):
            classdict= {"deer":1, "boar":2, "fox":3, "hare":4}
        elif(datasettype=="BayWald"):
            classdict= {"roe deer":1, "red deer":2}
        elif(datasettype=="Wildpark"):
            classdict= {"red deer":1, "fallow deer":2} 
        # for splitting attribut values, insert the new columns
        csvdata['track'] = pd.Series(np.random.randn(csvdata.shape[0]), index=csvdata.index)
        csvdata.insert(6,"ypoints",pd.Series(np.random.randn(csvdata.shape[0]), index=csvdata.index))
        # for giving each image file a unique image_id
        csvdata.insert(3,"image_id",pd.Series(np.random.randn(csvdata.shape[0]), index=csvdata.index))
        # groupby the filenames for generating a dictionary of corresponding image ids
        filegroup= csvdata.groupby(csvdata["filename"])
        num=np.arange(filegroup.ngroups)  
        imgid_dict= dict(zip(filegroup.groups.keys(),num))
        

        # preprocess the data from the csv for better reading from the dataframe
        for i in range(csvdata.shape[0]):
            # the if case is just interesting for datasets where files are that do not contain annotations
            if(int(csvdata.loc[i,"region_count"])>0):
                # write just the Video number int in the row for better accessing of the values
                p=csvdata.loc[i, "video_number"]
                val=[int(s) for s in p.split("\"") if s.isdigit()]
                #print(i)
                csvdata.loc[i, "video_number"]=val[0]            
                s=csvdata.loc[i,"xpoints"]
                sp= s.split("[")
    
                x_points= sp[1].split("]")[0]
                y_points= sp[2].split("]")[0]
                # concatenate the x and y points by just a ; for easier extraction later on 
            
                csvdata.loc[i,"xpoints"]=x_points
                csvdata.loc[i,"ypoints"]=y_points
                
                #prepare the region attributes column for better usage
                r=csvdata.loc[i,"class"]
                rs=r.split("\"")
    
                csvdata.loc[i,"class"]= classdict[rs[3]]
                csvdata.loc[i,"track"]=int(rs[7])
                #print(self.csvdata.loc[i,"region_attributes"])
                
                # insert image ids
                csvdata.loc[i,"image_id"] =int(imgid_dict[csvdata.loc[i,"filename"]])

        # filter out the rows where are no annotations
        csvdata = csvdata[csvdata["region_count"] !=0]      
        savename=annotationfile.split(".")[0]+"_preprocessed.csv"
        savepath=os.path.join(path,savename)
        csvdata.to_csv(savepath, index=False)
