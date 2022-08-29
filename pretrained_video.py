
import cv2 
import numpy as np
cap=cv2.VideoCapture(0)

while True:
    ret,frame=cap.read()
    frame_width=frame.shape[1]
    frame_height=frame.shape[0]
    frame_blob=cv2.dnn.blobFromImage(frame,1/255,(416,416),swapRB=True,crop=False)


    labels=["person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
                        "trafficlight","firehydrant","stopsign","parkingmeter","bench","bird","cat",
                        "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
                        "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sportsball",
                        "kite","baseballbat","baseballglove","skateboard","surfboard","tennisracket",
                        "bottle","wineglass","cup","fork","knife","spoon","bowl","banana","apple",
                        "sandwich","orange","broccoli","carrot","hotdog","pizza","donut","cake","chair",
                        "sofa","pottedplant","bed","diningtable","toilet","tvmonitor","laptop","mouse",
                        "remote","keyboard","cellphone","microwave","oven","toaster","sink","refrigerator",
                        "book","clock","vase","scissors","teddybear","hairdrier","toothbrush"]
    colors=np.random.uniform(0,255,size=(len(labels),3))



    model=cv2.dnn.readNetFromDarknet("pretrained_model/yolov3.cfg", "pretrained_model/yolov3.weights")  #cnfg dosyasını
    layers=model.getLayerNames()  
    
    output_layers=[layers[layer-1] for layer in model.getUnconnectedOutLayers()]   
    model.setInput(frame_blob)
    detection_layers=model.forward(output_layers)
    idsList=[]
    boxesList=[]
    confidenceList=[]


    for detection_layer in detection_layers:
        for object_detection in detection_layer:
            scores=object_detection[5:]  
            predicted_id=np.argmax(scores)  
            confidence=scores[predicted_id]  
            
            if confidence > 0.30:
                label=labels[predicted_id]
          
                bounding_box=object_detection[0:4] * np.array([frame_width,frame_height,frame_width,frame_height]) 
                
                (box_center_x,box_center_y,box_width,box_height)=bounding_box.astype("int")
                start_x=int(box_center_x - (box_width/2))
                start_y=int(box_center_y - (box_height/2))
             
                 
                idsList.append(predicted_id)
                confidenceList.append(float(confidence))
                boxesList.append([start_x,start_y,int(box_width),int(box_height)])
                
               
                
    
    maxids=cv2.dnn.NMSBoxes(boxesList,confidenceList,0.5,0.4)
    for maxid in maxids:
        maxClassID=maxid
        box=boxesList[maxClassID]
        start_x=box[0]
        start_y=box[1]
        box_width=box[2]
        box_height=box[3]
        predicted_id=idsList[maxClassID]
        label=labels[predicted_id]
        confidence=confidenceList[maxClassID]
        
    
            
        end_x=start_x + box_width
        end_y=start_y + box_height
        
        box_color=colors[predicted_id]
                    
                    
        label="{}: {:.2f}%".format(label, confidence*100) 
        print("Predicted object {}".format(label)) 
        cv2.rectangle(frame,(start_x,start_y),(end_x,end_y),box_color,2)
        cv2.putText(frame,label,(start_x,start_y-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,box_color,1)
            
        
        
    cv2.imshow("Detection Window",frame) 
    
    if cv2.waitKey(3) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
                  
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            


