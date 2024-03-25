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

# Function to process images in a folder
def process_images(input_folder):
    segmented_images = []
    for filename in os.listdir(input_folder):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            input_image_path = os.path.join(input_folder, filename)
            image = load_image(input_image_path)
            image_processed = pre_process(image)
            segmented_image = segment_tumor(image_processed)
            segmented_images.append(segmented_image)
    return segmented_images

# Function to train a random forest classifier
def train_random_forest(segmented_images, mask_images_arr):
    print("Training Random forest classifier...")
    seg_num_samples, seg_height, seg_width = segmented_images.shape
    segmented_images = segmented_images.reshape(seg_num_samples, seg_height * seg_width)
    mask_images_flat = np.array([mask.ravel() for mask in mask_images_arr])
    
    X_train, X_test, y_train, y_test = train_test_split(segmented_images, mask_images_flat, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    print("Model Trained.")
    return clf

# Function to save trained model
def save_model(model, output_folder):
    model_path = os.path.join(output_folder, 'random_forest_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved at {model_path}")

# Main function
if __name__ == "__main__":
    input_folder = "C:/Users/user2/OneDrive/Desktop/FINALYR_PROJECT/PROJECT_DATASET/PROJECT_RANDOM_FOREST/Intersection_code/DATA/p4/training"
    mask_folder  = "C:/Users/user2/OneDrive/Desktop/FINALYR_PROJECT/PROJECT_DATASET/PROJECT_RANDOM_FOREST/Intersection_code/DATA/p4/mask"
    model_output_folder = "C:/Users/user2/OneDrive/Desktop/FINALYR_PROJECT/PROJECT_DATASET/PROJECT_RANDOM_FOREST/Intersection_code/DATA"

    # Load mask images
    mask_images_arr = []
    for filenames in os.listdir(mask_folder):
        if filenames.endswith(".png") or filenames.endswith(".jpg"):
            mask_image_path = os.path.join(mask_folder, filenames)
            mask_image = load_image(mask_image_path)
            mask_images_arr.append(mask_image)

    # Process images
    segmented_images = process_images(input_folder)
    segmented_images_arr = np.array(segmented_images)
    mask_images_arr = np.array(mask_images_arr)

    # Train random forest classifier
    clf = train_random_forest(segmented_images_arr, mask_images_arr)

    # Save trained model
    save_model(clf, model_output_folder)
