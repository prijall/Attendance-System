import os, cv2
from sklearn.model_selection import train_test_split  #for separating data into train, test set

#@ Directory paths:
base_dir='Dataset'
train_dir=os.path.join(base_dir, 'train')
test_dir=os.path.join(base_dir, 'test')

#@ Creating Folders:
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

#@ Functions for capturing images:
def capture_images(name, user_id, nimgs=300):
    user_label=f'{name}_{user_id}'
    user_images_path=os.path.join(base_dir, 'raw', user_label) #storing 300 imgs in one file before splitting into train_test so that 
                                                               #data consistency is maintained
    os.makedirs(user_images_path, exist_ok=True)

    cap=cv2.VideoCapture(0) #default camera
    count=0

    while count<=nimgs:
        _, frame=cap.read()  
        cv2.putText(frame, f'Images Captured: {count}/{nimgs}', (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA) 
        cv2.imshow(f'Capturing images for {name} (press to quit)', frame)
       
        image_path=os.path.join(user_images_path, f'{user_label}_{count+1}.jpg')
        cv2.imwrite(image_path, frame)
        count+=1

        #for quiting:
        if cv2.waitKey(1)==ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return user_images_path
        
#@ Function to split image into train-test:
def split_dataset(user_images_path, train_ratio=0.8):
    image_files=os.listdir(user_images_path) #list of all images
    train_files, test_files=train_test_split(image_files, train_size=train_ratio, random_state=44)

    user_label=os.path.basename(user_images_path)
    user_train_dir=os.path.join(train_dir, user_label)
    user_test_dir=os.path.join(test_dir, user_label)
    os.makedirs(user_train_dir, exist_ok=True)
    os.makedirs(user_test_dir, exist_ok=True)
   
   #for moving files:
    for file in train_files:
        src=os.path.join(user_images_path, file)
        dst=os.path.join(user_train_dir, file)
        os.rename(src, dst)

    for file in test_files:
        src=os.path.join(user_images_path, file)
        dst=os.path.join(user_test_dir, file)
        os.rename(src, dst)


# main program:
if __name__=='__main__':
    name=input('Enter Employee name:')
    user_id=input('Enter Employee ID:')

    user_images_path=capture_images(name, user_id)
    