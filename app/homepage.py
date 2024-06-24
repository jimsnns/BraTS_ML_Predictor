import tensorflow as tf
from keras.models import load_model
from keras import backend as K
import segmentation_models_3D as sm
import nibabel as nib
import numpy as np
from io import BytesIO
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
import matplotlib.patches as mpatches
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
from matplotlib import pyplot as plt
from keras.metrics import MeanIoU, accuracy

st.set_page_config(
    page_title='MRI BraTS Predictor',
    page_icon='brats_webapp_image.jfif',
    layout="wide",
    initial_sidebar_state="expanded",
)

st.sidebar.image('brats_webapp_image.jfif')


def main():
    st.title("Welcome to MRI Brain Tumor Segmentation Application :brain:")

    instructions = """
    ##### This web application is designed for accessing and analyzing BraTS multimodal imaging data.
    ##### It provides access to multimodal brain tumor imaging scans available in NIfTI format (.nii.gz). If the file format is different the application won't work, as the model is designed and built upon a specific file type. 
    ##### The data you choose must have the following types of scans:

    - Post-contrast T1-weighted (T1CE)
    - T2-weighted (T2)
    - T2 Fluid Attenuated Inversion Recovery (T2-FLAIR)
    - Segmentation files which includes the annotated areas (SEG)
    ##### The segmantation files, are manual segmentations approved by experienced neuro-radiologists and they consist of the following labels:

    - GD-enhancing tumor (ET — label 3)
    - Peritumoral edema (ED — label 2)
    - Necrotic and non-enhancing tumor core (NCR/NET — label 1) \n
    These labels presented on the legend after the process of the uploaded files, and distincts the different labels with different colours.
    ##### Please upload all the required files, as per instructions above, by using the sidebar uploader. Then click on 'Process Files' button to see the prediction.
    """

    def generate_instr():
        for char in instructions:
            yield char


    # # Use st.write with a generator function to show title and instructions character by character
    st.write(''.join(generate_instr()))




    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    ## Loss setup
    wt0, wt1, wt2, wt3 = 0.26, 22.53, 22.53, 26.21
    dice_loss = sm.losses.DiceLoss()
    focal_loss = sm.losses.CategoricalFocalLoss()
    # iou_loss = keras_cv.losses.IoULoss()

    ## Metrics setup
    f_score = sm.metrics.FScore()  # Dice coefficient from segmentation models
    recall = tf.keras.metrics.Recall()
    mean_iou = tf.keras.metrics.MeanIoU(num_classes=4)
    iou = sm.metrics.IOUScore()

    losses = ['categorical_crossentropy', dice_loss, focal_loss]
    metrics = ['accuracy', iou, precision, recall, f_score, dice_loss, focal_loss]

    if "slice_number" not in st.session_state:
        st.session_state.slice_number = 64
    if "process_files" not in st.session_state:
        st.session_state.process_files = False

    def on_change_slider():
        st.session_state.process_files = True

    st.sidebar.title("Upload Files")
  
    file_uploaders = { 
        "T2": st.sidebar.file_uploader("Upload T2 file :file_folder:", type=["t2.nii", ".nii"]),
        "T1CE": st.sidebar.file_uploader("Upload T1CE file :file_folder:", type=["t1ce.nii", ".nii"]),
        "FLAIR": st.sidebar.file_uploader("Upload FLAIR file :file_folder:", type=["flair.nii", ".nii"]),
        "SEG": st.sidebar.file_uploader("Upload SEG file :file_folder:", type=["seg.nii", ".nii"]),
    }
    submit_button = st.button("Process Files")

    if st.session_state.process_files or submit_button:
        required_labels = ["T2", "T1CE", "FLAIR", "SEG"]
        uploaded_files = {}
        for file_label, file in file_uploaders.items():
            if file is not None:
                if file_label == 'T2' and file.name.endswith(("t2.nii")):
                    st.write(f"File {file_label} uploaded successfully! :white_check_mark:")
                    uploaded_files[file_label] = load_and_scale_uploaded_file(file,scaler)
                elif file_label == 'T1CE' and file.name.endswith(("t1ce.nii")):
                    st.write(f"File {file_label} uploaded successfully! :white_check_mark:")
                    uploaded_files[file_label] = load_and_scale_uploaded_file(file,scaler)
                elif file_label == 'FLAIR' and file.name.endswith(("flair.nii")):
                    st.write(f"File {file_label} uploaded successfully! :white_check_mark:")
                    uploaded_files[file_label] = load_and_scale_uploaded_file(file,scaler)
                elif file_label == 'SEG' and file.name.endswith(("seg.nii")):
                    st.write(f"File {file_label} uploaded successfully! :white_check_mark:")
                    st.write('\n\n')  
                    uploaded_files[file_label] = load_uploaded_file(file)
                else:
                    st.error(f"Invalid file type for {file_label}. Please upload a {file_label} NIfTI file. :x:")         
            else:
                st.error(f"Please upload a file for {file_label}. :x:")
                st.stop()

            # Ensure all required files are uploaded
        if all(label in uploaded_files for label in required_labels):
            # Stack the T2, T1CE, and FLAIR images along a new axis
            t2_data = uploaded_files["T2"]
            t1ce_data = uploaded_files["T1CE"]
            flair_data = uploaded_files["FLAIR"]
            seg_data = uploaded_files["SEG"]
            seg_data = seg_data.astype(np.uint8)
            seg_data[seg_data == 4] = 3

            combined_data = np.stack([t2_data, t1ce_data, flair_data], axis=3)
            combined_data = combined_data[56:184, 56:184, 13:141]
            seg_data = seg_data[56:184, 56:184, 13:141]

            ## For testing reasons, uncomment if needed   
            # st.write(f"Combined data shape: {combined_data.shape}")
            # st.write(f"Segmentation data shape: {seg_data.shape}")

            # Check the unique values in segmentation data
            unique_values = np.unique(seg_data)
            val, counts = np.unique(seg_data, return_counts=True) 

            ## For testing reasons, uncomment if needed 
            # st.write(f"Unique values in segmentation data: {val, counts}")

            temp_mask = tf.keras.utils.to_categorical(seg_data, num_classes=4)
            # Save combined data and segmentation mask as .npy files
            combined_file = BytesIO()
            np.save(combined_file, combined_data)
            combined_file.seek(0)  # Go to the start of the BytesIO object
            mask_file = BytesIO()
            np.save(mask_file, seg_data)
            mask_file.seek(0)  # Go to the start of the BytesIO object

            # Load the .npy files back to numpy arrays for further processing
            combined_data_np = np.load(combined_file)
            seg_data_np = np.load(mask_file)

            ## For testing reasons, uncomment if needed 
            # st.write(f"Loaded combined data shape: {combined_data_np.shape}")
            # st.write(f"Loaded segmentation data shape: {seg_data_np.shape}")

            # Load the .h5 from the ML model training.
            my_model = load_model('training_3d_200epoch_final_default.hdf5', 
                              custom_objects={'iou_score': iou,
                                              'dice_loss': sm.losses.DiceLoss(class_weights=np.array([wt0, wt1, wt2, wt3])),
                                              'focal_loss': focal_loss,
                                              'f1-score': f_score,
                                              'precision': precision})

            ## For testing reasons, uncomment if needed 
            # st.write(f"Temp mask is {temp_mask.shape}")
            test_seg_argmax = np.argmax(temp_mask, axis=3)
            test_combined_data = np.expand_dims(combined_data_np, axis=0) 
            test_prediction = my_model.predict(test_combined_data)
            test_prediction_argmax = np.argmax(test_prediction, axis=4)[0, :, :, :]

            # Assuming these are the class names
            class_names = ['Not Tumor', 'Non-Enhancing Tumor class 1', 'Edema class 2', 'Enhancing Tumor class 3']  # Add your actual class names
            # Assuming these are the colors used in your segmentation
            class_colors = ['#440054', '#3b528b', '#18b880', '#e6d74f']  # Adjust colors as per your class

            # Plot individual slices from test predictions for verification
            # n_slice = 80

            col1, col2, col3,col4 = st.columns(4)

             

            def display_slices(slice_number):
                with col1:
                    st.write("Testing Image (MRI Scan)")
                    fig, ax = plt.subplots()
                    ax.imshow(combined_data_np[:, :, slice_number, 0], cmap='gray')
                    st.pyplot(fig)
                    st.caption("### This is the real MRI scan which shows the basic information per sample.")

                with col2:
                    st.write("Ground Truth")
                    fig, ax = plt.subplots()
                    ax.imshow(test_seg_argmax[:, :, slice_number])
                    st.pyplot(fig)
                    st.caption("### This is the ground truth segmentation, with the annotations as created from experienced radiologists.")


                with col3:
                    st.write("Prediction on Test Image")
                    fig, ax = plt.subplots()
                    ax.imshow(test_prediction_argmax[:, :, slice_number])
                    st.pyplot(fig)
                    st.caption("### This is the predicted segmentation as result of our trained machine learning model.")


                with col4:
                # Create custom legend
                    st.write("Prediction Legend")
                    fig_legend, ax_legend = plt.subplots(figsize=(1 ,1))  # Increased size for better visibility
                    legend_patches = [mpatches.Patch(color=color, label=label) for color, label in zip(class_colors, class_names)]
                    ax_legend.legend(handles=legend_patches, loc='best', fontsize='xx-large')  # Adjust font size for better readability
                    ax_legend.axis('off')
                    st.pyplot(fig_legend)
                    
            # Custom CSS for the slider label
            st.markdown("""
                <style>
                .slider_label {
                    font-size: 20px;
                    font-weight: bold;
                }
                </style>
                """, unsafe_allow_html=True)
            
            # Display custom slider label
            st.markdown('<div class="slider_label">Slice Number</div>', unsafe_allow_html=True)
            st.write("\nChoose the desired slice number to see the prediction of the application.")
            slice_number = st.slider('',0, combined_data_np.shape[2] - 1, 100,on_change=on_change_slider)
            
            
            
            display_slices(slice_number)
            st.session_state.process_files = False

                

def load_uploaded_file(uploaded_file):
    # Read the file content into a BytesIO object
    file_content = uploaded_file.read()
    file_io = BytesIO(file_content)
    # Create a FileHolder and a FileMap for nibabel
    fileholder = nib.FileHolder(fileobj=file_io)
    img = nib.Nifti1Image.from_file_map({'header': fileholder, 'image': fileholder})
    return img.get_fdata()

# Custom function for T2, T1CE, FLAIR as we want the fit transform
def load_and_scale_uploaded_file(uploaded_file, scaler):
    file_content = uploaded_file.read()
    file_io = BytesIO(file_content)
    fileholder = nib.FileHolder(fileobj=file_io)
    img = nib.Nifti1Image.from_file_map({'header': fileholder, 'image': fileholder})
    img_data = img.get_fdata()
    # Scale the image data
    img_data_scaled = scaler.fit_transform(img_data.reshape(-1, img_data.shape[-1])).reshape(img_data.shape)
    return img_data_scaled

if __name__ == '__main__':
    main()
