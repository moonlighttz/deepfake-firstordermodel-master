import imageio
import skimage.transform
import numpy as np
import PIL.Image
import requests
from tqdm import tqdm
import os
import ffmpeg
import cv2
import io
from demo import load_checkpoints, make_animation  # Ensure this import works based on your directory structure
import ipywidgets as widgets
from IPython.display import display, HTML
import warnings
from skimage import img_as_ubyte
from tempfile import NamedTemporaryFile
from shutil import copyfileobj

warnings.filterwarnings("ignore")
os.makedirs("user", exist_ok=True)

def thumbnail(file):
    return imageio.get_reader(file, mode='I', format='FFMPEG').get_next_data()

def create_image(i, j):
    image_widget = widgets.Image.from_file(f'demo/images/{i}{j}.png')
    image_widget.add_class('resource')
    image_widget.add_class(f'resource-image{i}{j}')
    return image_widget

def create_video(i):
    video_widget = widgets.Image(
        value=cv2.imencode('.png', cv2.cvtColor(thumbnail(f'demo/videos/{i}.mp4'), cv2.COLOR_RGB2BGR))[1].tobytes(),
        format='png'
    )
    video_widget.add_class('resource')
    video_widget.add_class(f'resource-video{i}')
    return video_widget

def create_title(title):
    title_widget = widgets.Label(title)
    title_widget.add_class('title')
    return title_widget

def download_output(button):
    complete.layout.display = 'none'
    loading.layout.display = ''
    files.download('output.mp4')
    loading.layout.display = 'none'
    complete.layout.display = ''

def convert_output(button):
    complete.layout.display = 'none'
    loading.layout.display = ''
    ffmpeg.input('output.mp4').output('scaled.mp4', vf='scale=1080x1080:flags=lanczos,pad=1920:1080:420:0').overwrite_output().run()
    files.download('scaled.mp4')
    loading.layout.display = 'none'
    complete.layout.display = ''

def back_to_main(button):
    complete.layout.display = 'none'
    main.layout.display = ''

def resize_image(image, size=(256, 256)):
    w, h = image.size
    d = min(w, h)
    r = ((w - d) // 2, (h - d) // 2, (w + d) // 2, (h + d) // 2)
    return image.resize(size, resample=PIL.Image.LANCZOS, box=r)

def upload_image(change):
    global selected_image
    for name, file_info in upload_input_image_button.value.items():
        content = file_info['content']
        if content is not None:
            selected_image = resize_image(PIL.Image.open(io.BytesIO(content)).convert("RGB"))
            input_image_widget.clear_output(wait=True)
            with input_image_widget:
                display(selected_image)
            input_image_widget.add_class('uploaded')
upload_input_image_button.observe(upload_image, names='value')

def upload_video(change):
    global selected_video
    for name, file_info in upload_input_video_button.value.items():
        content = file_info['content']
        if content is not None:
            selected_video = f'user/{name}'
            with open(selected_video, 'wb') as video:
                video.write(content)
            preview = resize_image(PIL.Image.fromarray(thumbnail(selected_video)).convert("RGB"))
            input_video_widget.clear_output(wait=True)
            with input_video_widget:
                display(preview)
            input_video_widget.add_class('uploaded')
upload_input_video_button.observe(upload_video, names='value')

def change_model(change):
    if model.value.startswith('vox'):
        warning.remove_class('warn')
    else:
        warning.add_class('warn')
model.observe(change_model, names='value')

def generate(button):
    main.layout.display = 'none'
    loading.layout.display = ''
    filename = model.value + ('' if model.value == 'fashion' else '-cpk') + '.pth.tar'
    if not os.path.isfile(filename):
        response = requests.get(f'https://github.com/graphemecluster/first-order-model-demo/releases/download/checkpoints/{filename}', stream=True)
        with progress_bar:
            with tqdm.wrapattr(response.raw, 'read', total=int(response.headers.get('Content-Length', 0)), unit='B', unit_scale=True, unit_divisor=1024) as raw:
                with open(filename, 'wb') as file:
                    copyfileobj(raw, file)
        progress_bar.clear_output()
    reader = imageio.get_reader(selected_video, mode='I', format='FFMPEG')
    fps = reader.get_meta_data()['fps']
    driving_video = [frame for frame in reader]
    generator, kp_detector = load_checkpoints(config_path=f'config/{model.value}-256.yaml', checkpoint_path=filename)
    with progress_bar:
        predictions = make_animation(
            skimage.transform.resize(np.asarray(selected_image), (256, 256)),
            [skimage.transform.resize(frame, (256, 256)) for frame in driving_video],
            generator,
            kp_detector,
            relative=relative.value,
            adapt_movement_scale=adapt_movement_scale.value
        )
    progress_bar.clear_output()
    imageio.mimsave('output.mp4', [img_as_ubyte(frame) for frame in predictions], fps=fps)
    try:
        with NamedTemporaryFile(suffix='.mp4') as output:
            ffmpeg.output(ffmpeg.input('output.mp4').video, ffmpeg.input(selected_video).audio, output.name, c='copy').run()
            with open('output.mp4', 'wb') as result:
                copyfileobj(output, result)
    except ffmpeg.Error:
        pass
    output_widget.clear_output(True)
    with output_widget:
        video_widget = widgets.Video.from_file('output.mp4', autoplay=False, loop=False)
        video_widget.add_class('video')
        video_widget.add_class('video-left')
        display(video_widget)
    comparison_widget.clear_output(True)
    with comparison_widget:
        video_widget = widgets.Video.from_file(selected_video, autoplay=False, loop=False, controls=False)
        video_widget.add_class('video')
        video_widget.add_class('video-right')
        display(video_widget)
    loading.layout.display = 'none'
    complete.layout.display = ''

# UI Elements
label_or = widgets.Label('or')
label_or.add_class('label-or')

image_titles = ['Peoples', 'Cartoons', 'Dolls', 'Game of Thrones', 'Statues']
image_lengths = [8, 4, 8, 9, 4]

image_tab = widgets.Tab()
image_tab.children = [widgets.HBox([create_image(i, j) for j in range(length)]) for i, length in enumerate(image_lengths)]
for i, title in enumerate(image_titles):
    image_tab.set_title(i, title)

input_image_widget = widgets.Output()
input_image_widget.add_class('input-widget')
upload_input_image_button = widgets.FileUpload(accept='image/*', button_style='primary')
upload_input_image_button.add_class('input-button')
image_part = widgets.HBox([
    widgets.VBox([input_image_widget, upload_input_image_button]),
    label_or,
    image_tab
])

video_tab = widgets.Tab()
video_tab.children = [widgets.HBox([create_video(i) for i in range(5)])]
video_tab.set_title(0, 'All Videos')

input_video_widget = widgets.Output()
input_video_widget.add_class('input-widget')
upload_input_video_button = widgets.FileUpload(accept='video/*', button_style='primary')
upload_input_video_button.add_class('input-button')
video_part = widgets.HBox([
    widgets.VBox([input_video_widget, upload_input_video_button]),
    label_or,
    video_tab
])

model = widgets.Dropdown(
    description="Model:",
    options=[
        'vox',
        'vox-adv',
        'taichi',
        'taichi-adv',
        'nemo',
        'mgif',
        'fashion',
        'bair'
    ]
)
warning = widgets.HTML('<b>Warning:</b> Upload your own images and videos (see README)')
warning.add_class('warning')
model_part = widgets.HBox([model, warning])

relative = widgets.Checkbox(description="Relative keypoint displacement (Inherit object proportions from the video)", value=True)
adapt_movement_scale = widgets.Checkbox(description="Adapt movement scale (Don’t touch unless you know want you are doing)", value=True)
generate_button = widgets.Button(description="Generate", button_style='primary')
main = widgets.VBox([
    create_title('Choose Image'),
    image_part,
    create_title('Choose Video'),
    video_part,
    create_title('Settings'),
    model_part,
    relative,
    adapt_movement_scale,
    generate_button
])

loader = widgets.Label()
loader.add_class("loader")
loading_label = widgets.Label("This may take several minutes to process…")
loading_label.add_class("loading-label")
progress_bar = widgets.Output()
loading = widgets.VBox([loader, loading_label, progress_bar])
loading.add_class('loading')

output_widget = widgets.Output()
output_widget.add_class('output-widget')
download = widgets.Button(description='Download', button_style='primary')
download.add_class('output-button')
download.on_click(download_output)
convert = widgets.Button(description='Convert to 1920×1080', button_style='primary')
convert.add_class('output-button')
convert.on_click(convert_output)
back = widgets.Button(description='Back', button_style='primary')
back.add_class('output-button')
back.on_click(back_to_main)

comparison_widget = widgets.Output()
comparison_widget.add_class('comparison-widget')
comparison_label = widgets.Label('Comparison')
comparison_label.add_class('comparison-label')
complete = widgets.HBox([
    widgets.VBox([output_widget, download, convert, back]),
    widgets.VBox([comparison_widget, comparison_label])
])

display(widgets.VBox([main, loading, complete]))

generate_button.on_click(generate)

loading.layout.display = 'none'
complete.layout.display = 'none'
