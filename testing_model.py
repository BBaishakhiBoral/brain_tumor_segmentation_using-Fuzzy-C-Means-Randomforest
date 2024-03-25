import os
import numpy as np
from skimage import io, color, filters
from sklearn.cluster import KMeans
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Function to load an image
def load_image(image_path): 
    return io.imread(image_path, as_gray=True)

# Function to preprocess an image
def pre_process(image):
    if len(image.shape) > 2:
        image = color.rgb2gray(image)
    image_blur = filters.gaussian(image, sigma=1)
    return image_blur


# Fuzzy C-means clustering algorithm
def fuzzy_c_means(data, n_clusters, m):
    def objective_function(centers, data):
        distances = np.linalg.norm(data[:, np.newaxis] - centers, axis=2)
        membership = 1 / distances ** (2 / (m - 1))
        membership_sum = np.sum(membership, axis=1)
        objective = np.sum((membership ** m) * distances)
        return objective

    initial_guess = np.random.rand(n_clusters * data.shape[-1])
    result = minimize(objective_function, initial_guess, args=(data,),
                      method='L-BFGS-B', options={'disp': False})
    return result.x.reshape(n_clusters, data.shape[-1])


# Function to segment tumor from an image
def segment_tumor(image, n_clusters=3, m=2):
    data = image.reshape(-1, 1)
    centers = fuzzy_c_means(data, n_clusters, m)
    kmeans = KMeans(n_clusters=n_clusters,
                    init=centers.reshape(-1, 1), n_init=1)
    kmeans.fit(data)
    segmented_image = kmeans.labels_.reshape(image.shape)
    return segmented_image

# Function to generate tumor mask using trained random forest classifier
def generate_tumor_mask(image, clf):
    image_flat = image.ravel()
    tumor_mask = clf.predict([image_flat])
    tumor_mask = tumor_mask.reshape(image.shape)
    return tumor_mask

# Function to load trained model
def load_model(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

# Function to save tumor mask image
def save_tumor_mask(tumor_mask, output_folder, image_index):
    output_path = os.path.join(output_folder, f"tumor_mask_{image_index}.png")
    io.imsave(output_path, tumor_mask)

# Main function
if __name__ == "__main__":
    input_folder = "C:/Users/user2/OneDrive/Desktop/FINALYR_PROJECT/PROJECT_DATASET/PROJECT_RANDOM_FOREST/Intersection_code/DATA/p4/training"
    model_path = "C:/Users/user2/OneDrive/Desktop/FINALYR_PROJECT/PROJECT_DATASET/PROJECT_RANDOM_FOREST/Intersection_code/DATA/random_forest_model.pkl"
    output_folder= "C:/Users/user2/OneDrive/Desktop/FINALYR_PROJECT/PROJECT_DATASET/PROJECT_RANDOM_FOREST/Intersection_code/DATA/output_masks"
    
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Load trained model
    clf = load_model(model_path)
    
    # Process each image in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            # Load input image
            input_image_path = os.path.join(input_folder, filename)
            image = load_image(input_image_path)
            
            # Preprocess input image
            image_processed = pre_process(image)
            
            # Segment tumor from preprocessed image
            segmented_image = segment_tumor(image_processed)
            
            # Generate tumor mask using trained model
            tumor_mask = generate_tumor_mask(segmented_image, clf)
            
            # Save tumor mask
            save_tumor_mask(tumor_mask, output_folder, filename.split('.')[0])  # Save using image filename without extension

