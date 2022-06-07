from PIL import Image
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import logging
import json
import argparse


def reload_model(model_path):
    model = tf.keras.models.load_model(model_path, custom_objects={'KerasLayer':hub.KerasLayer})
    return model

def predict(image_path, model, top_k):
    image = Image.open(image_path)
    test_image = np.asarray(image)
    processed_image = process_image(test_image)
    prediction = model.predict(np.expand_dims(processed_image, axis=0))
    top_values, top_indices = tf.math.top_k(prediction, top_k)
    
    return top_values.numpy(), top_indices.numpy()

def process_image(image):
    image_size = 224
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    image = tf.image.resize(image, (image_size, image_size))
    image /= 255
    return image.numpy()


if __name__ == '__main__':

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    
    parser = argparse.ArgumentParser(description='Informaton about the image')

    parser.add_argument('image_path', type=str, help='Path to the image')
    parser.add_argument('model', type=str, help='Path to the model')
    parser.add_argument('--top_k', type=int, help='Number of top predictions')
    parser.add_argument('--category_names', type=str, help='Path to the category names json file')

    args = parser.parse_args()
    parser.error('Image path is required') if args.image_path is None else None
    parser.error('Model path is required') if args.model is None else None

    if (args.top_k is None):
        args.top_k = 1
    
    cat_to_name = None
    if (args.category_names is not None):
        with open(args.category_names, 'r') as f:
            cat_to_name = json.load(f)
    
    model = reload_model(args.model)

    logger.info('Predicting')

    top_values, top_indices = predict(args.image_path, model, args.top_k)

    for i in range(args.top_k):
        if (cat_to_name is not None):
            logger.info(f'{i+1}. {cat_to_name[str(top_indices[0][i])]} {top_values[0][i]:.5f}')
        else:
            logger.info(f'{i+1}. {top_indices[0][i]} {top_values[0][i]:.5f}')
    
    logger.info('Done')


