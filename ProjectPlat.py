import cv2
import easyocr
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.filechooser import FileChooserListView
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.image import Image as KivyImage
from kivy.graphics.texture import Texture

class ImageApp(App):
    def build(unsharp):
        unsharp.layout = BoxLayout(orientation='vertical')

        unsharp.label = Label(text="Select an image to process", size_hint=(1, 0.1))
        unsharp.layout.add_widget(unsharp.label)

        unsharp.file_chooser = FileChooserListView(
            filters=['*.jpg', '*.jpeg', '*.png'],
            path="sample_images" # Edit this path if your image folder is located elsewhere.
        )
        unsharp.layout.add_widget(unsharp.file_chooser)

        unsharp.process_button = Button(
            text="Process Image",
            size_hint=(0.3, 0.1),
            pos_hint={"center_x": 0.5}
        )
        unsharp.process_button.bind(on_press=unsharp.process_image)
        unsharp.layout.add_widget(unsharp.process_button)

        image_layout = BoxLayout(orientation='horizontal', size_hint=(1, 0.6))
        unsharp.original_image_widget = KivyImage(size_hint=(0.5, 1))  
        unsharp.plate_image_widget = KivyImage(size_hint=(0.5, 1))  
        image_layout.add_widget(unsharp.original_image_widget)
        image_layout.add_widget(unsharp.plate_image_widget)
        unsharp.layout.add_widget(image_layout)

        return unsharp.layout

    def process_image(unsharp, instance):
        image_path = unsharp.file_chooser.selection
        if not image_path:
            unsharp.label.text = "No image selected. Please try again."
            return

        image_path = image_path[0]
        unsharp.label.text = f"Processing: {image_path}"

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        gaussian_blur = cv2.GaussianBlur(gray, (9, 9), 10.0)
        unsharp_image = cv2.addWeighted(gray, 1.5, gaussian_blur, -0.5, 0)

        edges = cv2.Canny(unsharp_image, 30, 200)
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

        plate_contour = None
        for contour in contours:
            approx = cv2.approxPolyDP(contour, 0.018 * cv2.arcLength(contour, True), True)
            if len(approx) == 4:
                plate_contour = approx
                break

        if plate_contour is not None:
            x, y, w, h = cv2.boundingRect(plate_contour)
            plate = unsharp_image[y:y + h, x:x + w]

            unsharp.show_image(image, unsharp.original_image_widget)
            unsharp.show_image(plate, unsharp.plate_image_widget, is_gray=True)

            reader = easyocr.Reader(['en'])
            result = reader.readtext(plate)
            detected_texts = [text[1] for text in result]
            unsharp.label.text = f"Detected Text: {' '.join(detected_texts)}"
        else:
            unsharp.label.text = "Number Plate Not Detected"

    def show_image(unsharp, img, widget, is_gray=False):
        if is_gray:
            buf = cv2.flip(img, 0).tobytes()
            texture = Texture.create(size=(img.shape[1], img.shape[0]), colorfmt='luminance')
            texture.blit_buffer(buf, colorfmt='luminance', bufferfmt='ubyte')
        else:
            buf = cv2.flip(img, 0).tobytes()
            texture = Texture.create(size=(img.shape[1], img.shape[0]), colorfmt='rgb')
            texture.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')
        texture.wrap = 'clamp_to_edge'
        widget.texture = texture


if __name__ == '__main__':
    ImageApp().run()
