import json
import sys
import glob
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import skimage
import os
import argparse
from detection.hi_sam.modeling.build import model_registry
from detection.hi_sam.modeling.auto_mask_generator import AutoMaskGenerator
from tqdm import tqdm
from PIL import Image
import random
from detection.utils import utilities
from shapely.geometry import Polygon
import pyclipper
import datetime
import warnings
import shutil
from parseq.strhub.data.module import SceneTextDataModule
from parseq.strhub.models.utils import load_from_checkpoint
import sort_para
from pdf2image import convert_from_path
from parseq.models import parseq_models
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
warnings.filterwarnings("ignore")

print('\n\n')
print("--------------------------------- Lipikar OCR Framework, IIT-Delhi ---------------------------------")
print('\n')


def get_args_parser():
    parser = argparse.ArgumentParser('Hi_SAM', add_help=False)

    parser.add_argument("--input", type=str, default='./test_pdf', nargs="+",
                        help="Path to the input image or PDF")
    parser.add_argument("--output", type=str, default='./demo',
                        help="A file or directory to save output visualizations.")
    parser.add_argument("--existing_fgmask_input", type=str, default='./datasets/HierText/val_fgmask/',
                        help="A file or directory of foreground masks.")
    parser.add_argument("--model-type", type=str, default="vit_l",
                        help="The type of model to load, e.g., 'vit_h', 'vit_l', 'vit_b', 'vit_s'")
    parser.add_argument("--checkpoint", type=str, default='./detection/pretrained_checkpoint/det_l.pth',
                        help="The path to the SAM checkpoint to use for mask generation.")
    parser.add_argument("--device", type=str, default="cuda",
                        help="The device to run generation on.")
    parser.add_argument("--hier_det", default=True)
    parser.add_argument("--use_fgmask", action='store_true')
    parser.add_argument("--vis", action='store_true')
    parser.add_argument("--eval", type=bool, default=True)
    parser.add_argument('--total_points', default=1500, type=int, help='The number of foreground points')
    parser.add_argument('--batch_points', default=100, type=int, help='The number of points per batch')

    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--input_size', default=[1024, 1024], type=list)
    parser.add_argument('--out_dir', default='results', type=str,
                        help='Directory to save detection outputs and word-level crops')

    # Self-prompting parameters
    parser.add_argument('--attn_layers', default=1, type=int,
                        help='The number of image-to-token cross-attention layers in model_aligner')
    parser.add_argument('--prompt_len', default=12, type=int, help='The number of prompt tokens')
    parser.add_argument('--layout_thresh', type=float, default=0.5)
    parser.add_argument('--lang', type=str, default='english')
    parser.add_argument('--save_crops', action='store_true')
    parser.add_argument('--output_type', type=str, default='text', choices=['text', 'pdf'],
                        help='Output type: "text" or "pdf"')
    return parser.parse_args()

# Register fonts for each language
pdfmetrics.registerFont(TTFont('Devanagari', '../fonts/NotoSansDevanagari-VariableFont_wdth,wght.ttf'))  # Hindi, Marathi, Konkani
pdfmetrics.registerFont(TTFont('NotoSansKashmiri', '../fonts/NotoSansDevanagari-VariableFont_wdth,wght.ttf'))  # Kashmiri
pdfmetrics.registerFont(TTFont('Maithili', '../fonts/NotoSansDevanagari-VariableFont_wdth,wght.ttf'))  # Maithili
pdfmetrics.registerFont(TTFont('NotoSansNepali', '../fonts/NotoSansDevanagari-VariableFont_wdth,wght.ttf'))  # Nepali
pdfmetrics.registerFont(TTFont('NotoSansBengali', '../fonts/NotoSansBengali-VariableFont_wdth,wght.ttf'))  # Bengali
pdfmetrics.registerFont(TTFont('NotoSansGujarati', '../fonts/NotoSansGujarati-VariableFont_wdth,wght.ttf'))  # Gujarati
pdfmetrics.registerFont(TTFont('NotoSansKannada', '../fonts/NotoSansKannada-VariableFont_wdth,wght.ttf'))  # Kannada
pdfmetrics.registerFont(TTFont('NotoSansMalayalam', '../fonts/NotoSansMalayalam-VariableFont_wdth,wght.ttf'))  # Malayalam
pdfmetrics.registerFont(TTFont('NotoSansPunjabi', '../fonts/NotoSansGurmukhi-VariableFont_wdth,wght.ttf'))  # Punjabi
pdfmetrics.registerFont(TTFont('NotoSansTamil', '../fonts/NotoSansTamil-VariableFont_wdth,wght.ttf'))  # Tamil
pdfmetrics.registerFont(TTFont('NotoSansOdia', '../fonts/NotoSansOriya-VariableFont_wdth,wght.ttf'))  # Odia
pdfmetrics.registerFont(TTFont('Helvetica', '../fonts/Helvetica.ttf'))
pdfmetrics.registerFont(TTFont('Marathi', '../fonts/TiroDevanagariMarathi-Regular.ttf'))
pdfmetrics.registerFont(TTFont('Arabic', '../fonts/NotoSansArabic-VariableFont_wdth,wght.ttf'))
# Map language codes to the appropriate font
def get_font_for_language(language):
    font_map = {
        'hindi': 'Devanagari',  # Hindi, Marathi, Konkani
        'kashmiri': 'NotoSansKashmiri',  # Kashmiri
        'maithili': 'Maithili',  # Maithili
        'nepali': 'NotoSansNepali',  # Nepali
        'bengali': 'NotoSansBengali',  # Bengali
        'gujarati': 'NotoSansGujarati',  # Gujarati
        'kannada': 'NotoSansKannada',  # Kannada
        'malayalam': 'NotoSansMalayalam',  # Malayalam
        'punjabi': 'NotoSansPunjabi',  # Punjabi
        'tamil': 'NotoSansTamil',  # Tamil
        'odia': 'NotoSansOdia',  # Odia
        'marathi' : 'Marathi',
        'kashmiri' : 'Devanagari',
        'konkani' : 'Devanagari',
        'english': 'Helvetica',  # English default font
    }
    return font_map.get(language, 'Helvetica')  # Default to Helvetica if the language isn't found

def show_points(coords, ax, marker_size=40):
    ax.scatter(coords[:, 0], coords[:, 1], color='green', marker='*',
               s=marker_size, edgecolor='white', linewidth=0.25)


def show_mask(mask, ax, random_color=False, color=None):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.5])], axis=0)
    else:
        color = color if color is not None else np.array(
            [30 / 255, 144 / 255, 255 / 255, 0.5])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_hi_masks(masks, filename, image):
    plt.figure(figsize=(15, 15), dpi=200)
    plt.imshow(image)

    for hi_mask in masks:
        hi_mask = hi_mask[0]
        show_mask(hi_mask, plt.gca(), random_color=True)
    plt.axis('off')
    plt.savefig(filename, bbox_inches='tight')
    plt.close()


def save_binary_mask(mask: np.array, filename):
    if len(mask.shape) == 3:
        assert mask.shape[0] == 1
        mask = mask[0].astype(np.uint8) * 255
    elif len(mask.shape) == 2:
        mask = mask.astype(np.uint8) * 255
    else:
        raise NotImplementedError
    mask = Image.fromarray(mask)
    mask.save(filename)


def unclip(p, unclip_ratio=2.0):
    poly = Polygon(p)
    distance = poly.area * unclip_ratio / poly.length
    offset = pyclipper.PyclipperOffset()
    offset.AddPath(
        p, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    expanded = np.array(offset.Execute(distance))
    return expanded


def find_image_path(folder_path, image_name):
    # List of common image file extensions
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']

    # Walk through all files in the folder and subfolders
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_name, file_ext = os.path.splitext(file)
            if file_name == image_name and file_ext.lower() in image_extensions:
                return os.path.join(root, file)

    return None  # Return None if the image is not found


def convert_pdf_to_images(pdf_path, output_folder):
    # Convert PDF pages to images
    images = convert_from_path(pdf_path)

    # Iterate through the images and save each one
    for i, image in enumerate(images):
        image_path = os.path.join(
            output_folder, 'page_' + str(i + 1).zfill(3) + '.png')
        image.save(image_path, 'PNG')


def calculate_font_size_for_box(text, box_width, box_height, font_name):
    face = pdfmetrics.getFont(font_name).face
    ascent = face.ascent
    descent = face.descent
    
    font_size = int((box_height * 1000) / (ascent - descent))

    text_width = pdfmetrics.stringWidth(text, font_name, font_size)
    
    if text_width > box_width:
        font_size = font_size * (box_width / text_width)
    
    return int(font_size)

if __name__ == '__main__':
    args = get_args_parser()
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Load the Hi-SAM model
    hisam = model_registry[args.model_type](args)
    hisam.eval()
    hisam.to(args.device)

    efficient_hisam = args.model_type in ['vit_s', 'vit_t']
    amg = AutoMaskGenerator(hisam, efficient_hisam=efficient_hisam)
    none_num = 0

    # Ensure output directories exist
    if os.path.exists(args.out_dir):
        shutil.rmtree(args.out_dir)
    os.makedirs(args.out_dir)
    json_out_dir = os.path.join(args.out_dir, 'detection_outputs')
    os.makedirs(json_out_dir, exist_ok=True)
    combined_json_dir = os.path.join(args.out_dir, 'combined_jsons')
    os.makedirs(combined_json_dir, exist_ok=True)

    # Initialize pdf_img_dir
    pdf_img_dir = 'input_images'
    if os.path.exists(pdf_img_dir):
        shutil.rmtree(pdf_img_dir)
    os.makedirs(pdf_img_dir)

    # List of supported file extensions
    supported_extensions = ['.pdf', '.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']

    # Initialize input_dirs to store directories containing images
    input_dirs = []

    if os.path.isdir(args.input[0]):
        input_image_dir = args.input[0]
        args.input = [os.path.join(input_image_dir, fname)
                      for fname in os.listdir(input_image_dir)
                      if os.path.isfile(os.path.join(input_image_dir, fname)) and
                      os.path.splitext(fname)[1].lower() in supported_extensions]
        if not args.input:
            print(f"No supported files found in directory: {input_image_dir}")
            sys.exit(1)
        if any(os.path.splitext(f)[1].lower() == '.pdf' for f in args.input):
            for pdf_path in args.input:
                if os.path.splitext(pdf_path)[1].lower() == '.pdf':
                    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
                    pdf_page_dir = os.path.join(pdf_img_dir, pdf_name)
                    if os.path.exists(pdf_page_dir):
                        shutil.rmtree(pdf_page_dir)
                    os.makedirs(pdf_page_dir)
                    convert_pdf_to_images(pdf_path, pdf_page_dir)
                    input_dirs.append(pdf_page_dir)
        else:
            # Handle image files in directory
            images_dir = os.path.join(pdf_img_dir, 'images')
            if os.path.exists(images_dir):
                shutil.rmtree(images_dir)
            os.makedirs(images_dir)
            for img_file in args.input:
                shutil.copy(img_file, images_dir)
            input_dirs.append(images_dir)
    else:
        # Handle the case where args.input[0] is a file
        if os.path.splitext(args.input[0])[1].lower() == '.pdf':
            pdf_path = args.input[0]
            pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
            pdf_page_dir = os.path.join(pdf_img_dir, pdf_name)
            if os.path.exists(pdf_page_dir):
                shutil.rmtree(pdf_page_dir)
            os.makedirs(pdf_page_dir)
            convert_pdf_to_images(pdf_path, pdf_page_dir)
            input_dirs.append(pdf_page_dir)
        elif os.path.splitext(args.input[0])[1].lower() in supported_extensions:
            # Handle single image file
            images_dir = os.path.join(pdf_img_dir, 'images')
            if os.path.exists(images_dir):
                shutil.rmtree(images_dir)
            os.makedirs(images_dir)
            for img_file in args.input:
                shutil.copy(img_file, images_dir)
            input_dirs.append(images_dir)
        else:
            print(f"Unsupported input format: {args.input[0]}")
            sys.exit(1)

    # Initialize progress tracking variables
    total_steps = 0  # Total steps for both detection and recognition
    processed_steps = 0

    # Calculate total number of pages or images for progress estimation
    total_pages = 0
    for pdf_images_dir in input_dirs:
        if not os.path.exists(pdf_images_dir):
            continue
        pdf_input_images = [os.path.join(pdf_images_dir, fname)
                            for fname in os.listdir(pdf_images_dir)
                            if os.path.isfile(os.path.join(pdf_images_dir, fname))]
        total_pages += len(pdf_input_images)

    # Assume detection and recognition each take equal time for simplicity
    total_steps = total_pages * 2  # Multiply by 2 for detection and recognition

    print("Detection started.")
    for pdf_idx, pdf_images_dir in enumerate(tqdm(input_dirs)):
        pdf_name = os.path.basename(pdf_images_dir)
        if not os.path.exists(pdf_images_dir):
            print(f"Error: Directory {pdf_images_dir} does not exist.")
            continue
        pdf_input_images = [os.path.join(pdf_images_dir, fname)
                            for fname in os.listdir(pdf_images_dir)
                            if os.path.isfile(os.path.join(pdf_images_dir, fname))]
        pdf_json_dir = os.path.join(json_out_dir, pdf_name)
        os.makedirs(pdf_json_dir, exist_ok=True)

        for page_idx, path in enumerate(pdf_input_images):
            img_id = os.path.basename(path).split('.')[0]
            if os.path.isdir(args.output):
                img_name = img_id + '.png'
                out_filename = os.path.join(args.output, img_name)
            else:
                out_filename = args.output

            image = cv2.imread(path)
            if image is None:
                print(f"Error: {path} could not be loaded.")
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img_h, img_w = image.shape[:2]

            if args.use_fgmask:
                fgmask_path = os.path.join(
                    args.existing_fgmask_input, img_id + '.png')
                fgmask = skimage.io.imread(fgmask_path)
                amg.set_fgmask(fgmask)

            amg.set_image(image)

            masks, scores, affinity = amg.predict(
                from_low_res=False,
                fg_points_num=args.total_points,
                batch_points_num=args.batch_points,
                score_thresh=0.5,
                nms_thresh=0.5,
            )

            if args.eval:
                if masks is None:
                    lines = [{'words': [{'text': '', 'vertices': [
                        [0, 0], [1, 0], [1, 1], [0, 1]]}], 'text': ''}]
                    paragraphs = [{'lines': lines}]
                    result = {
                        'image_id': img_id,
                        "paragraphs": paragraphs
                    }
                    none_num += 1
                else:
                    masks = (masks[:, 0, :, :]).astype(np.uint8)
                    lines = []
                    line_indices = []
                    for index, mask in enumerate(masks):
                        line = {'words': [], 'text': ''}
                        contours, _ = cv2.findContours(
                            mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                        for cont in contours:
                            epsilon = 0.002 * cv2.arcLength(cont, True)
                            approx = cv2.approxPolyDP(cont, epsilon, True)
                            points = approx.reshape((-1, 2))
                            if points.shape[0] < 4:
                                continue
                            pts = unclip(points)
                            if len(pts) != 1:
                                continue
                            pts = pts[0].astype(np.int32)
                            if Polygon(pts).area < 32:
                                continue
                            pts[:, 0] = np.clip(pts[:, 0], 0, img_w)
                            pts[:, 1] = np.clip(pts[:, 1], 0, img_h)
                            cnt_list = pts.tolist()
                            line['words'].append(
                                {'text': '', 'vertices': cnt_list})
                        if line['words']:
                            lines.append(line)
                            line_indices.append(index)

                    line_grouping = utilities.DisjointSet(len(line_indices))
                    if len(line_indices) > 0:
                        affinity = affinity[line_indices][:, line_indices]
                        for i1, i2 in zip(*np.where(affinity > args.layout_thresh)):
                            line_grouping.union(i1, i2)
                        line_groups = line_grouping.to_group()
                    else:
                        line_groups = []
                    paragraphs = []
                    for line_group in line_groups:
                        paragraph = {'lines': []}
                        for id_ in line_group:
                            paragraph['lines'].append(lines[id_])
                        if paragraph:
                            paragraphs.append(paragraph)
                    result = {
                        'image_id': img_id,
                        "paragraphs": paragraphs
                    }

                with open(os.path.join(pdf_json_dir, img_id + '.jsonl'), 'w', encoding='utf-8') as fw:
                    json.dump(result, fw)

                # Increment processed steps and print progress
                processed_steps += 1
                progress = int((processed_steps / total_steps) * 100)
                print(f'PROGRESS:{progress}')
                sys.stdout.flush()

        jsonl_list = glob.glob(pdf_json_dir + '/*.jsonl')
        jsonl_list.sort()
        final_results = {"annotations": []}
        for jsonl_name in jsonl_list:
            with open(jsonl_name, 'r') as fr:
                res = json.load(fr)
            final_results['annotations'].append(res)

        with open(os.path.join(combined_json_dir, pdf_name + '.jsonl'), 'w') as fw:
            json.dump(final_results, fw, ensure_ascii=False)

    print("Detection complete.")
    del hisam
    del amg
    torch.cuda.empty_cache()

    def get_checkpoint_path():
        # Check if the script is running as a PyInstaller bundle
        if getattr(sys, 'frozen', False):
            base_path = sys._MEIPASS
        else:
            base_path = os.path.abspath(".")

        # Path to the checkpoint within the bundle
        return os.path.join(base_path, parseq_models[args.lang])

    parseq = load_from_checkpoint(get_checkpoint_path()).eval().to(args.device)
    img_transform = SceneTextDataModule.get_transform(
        parseq.hparams.img_size)

    out_crops_dir = os.path.join(args.out_dir, 'crops')

    recog_dir = os.path.join(args.out_dir, 'recognition_results_txt')
    os.makedirs(recog_dir, exist_ok=True)
    out_pdf_dir = os.path.join(args.out_dir, 'recognition_results_pdf')
    os.makedirs(out_pdf_dir, exist_ok=True)

    print("Recognition started.")
    for pdf_idx, pdf_images_dir in enumerate(tqdm(input_dirs)):
        pdf_name = os.path.basename(pdf_images_dir)
        txt_output = os.path.join(recog_dir, pdf_name + '.txt')
        combined_json_path = os.path.join(combined_json_dir, pdf_name + '.jsonl')
        if not os.path.exists(combined_json_path):
            print(f"Error: Combined JSON file {combined_json_path} does not exist.")
            continue
        with open(combined_json_path, 'r') as fr:
            res = json.load(fr)
        reordered_data = sort_para.reorder_json_data(res)

        if args.output_type == 'pdf':
            output_pdf_path = os.path.join(out_pdf_dir, pdf_name + '.pdf')
            c = canvas.Canvas(output_pdf_path, pagesize=letter)
            page_width, page_height = letter
        else:
            c = None  # No need to create a canvas for text output

        with open(txt_output, 'w') as f:
            for annotation in reordered_data['annotations']:
                image_id = annotation['image_id']
                f.write(image_id + '\n')
                f.write('-------------------------------------------------------------------------------------------------\n')

                if args.save_crops:
                    annot_path = os.path.join(out_crops_dir, image_id)
                    os.makedirs(annot_path, exist_ok=True)

                image_path = find_image_path(
                    pdf_images_dir, image_id)

                image = cv2.imread(image_path)
                if image is None:
                    print(f"Error: Image {image_id} could not be loaded.")
                    continue
                if args.output_type == 'pdf':
                    image_height, image_width = image.shape[:2]
                    scale_width = page_width / image_width
                    scale_height = page_height / image_height

                    scale = min(scale_height, scale_width)
                    new_width = image_width * scale
                    new_height = image_height * scale
                    x_position = (page_width - new_width) / 2
                    y_position = (page_height - new_height) / 2

                for p, para in enumerate(annotation['paragraphs']):
                    for l, line in enumerate(para['lines']):
                        line_str = ''
                        # font_size = 16
                        line_vertices = []
                        x_coord = y_coord = 0  # Initialize x_coord and y_coord
                        for wd, word in enumerate(line['words']):
                            vertices = word['vertices']
                            line_vertices += vertices
                            pts = np.array(vertices, np.int32)

                            rect = cv2.minAreaRect(pts)
                            box = cv2.boxPoints(rect)
                            box = np.intp(box)
                            mask = np.zeros_like(image)
                            cv2.fillPoly(mask, [box], (255, 255, 255))
                            cropped = cv2.bitwise_and(image, mask)
                            x, y, w, h = cv2.boundingRect(box)
                            x = max(x, 0)
                            y = max(y, 0)
                            cropped_image = cropped[y:y + h, x:x + w]

                            if cropped_image.size == 0:
                                continue

                            img = Image.fromarray(cropped_image)
                            img = img_transform(
                                img).unsqueeze(0).to(args.device)

                            logits = parseq(img)

                            pred = logits.softmax(-1)
                            label, confidence = parseq.tokenizer.decode(pred)
                            if wd == 0:
                                line_str += label[0]
                            else:
                                line_str += ' ' + label[0]
                            # font_size_word = rect[1][1]
                            # font_size = min(font_size, font_size_word)
                            if wd == 0 and c is not None:
                                # x_coord = (x / image_width) * page_width
                                # y_coord = (1 - y / image_height) * page_height
                                x_coord = x*scale + x_position
                                y_coord = page_height - ((y+h)*scale + y_position)

                            if args.save_crops:
                                image_out_name = f"p{str(p).zfill(2)}_l{str(l).zfill(2)}_w{str(wd).zfill(2)}.png"
                                image_out_path = os.path.join(
                                    out_crops_dir, annot_path, image_out_name)
                                cv2.imwrite(image_out_path, cropped_image)

                        f.write(line_str + '\n')

                        if args.output_type == 'pdf' and c is not None and line_str.strip():
                            line_vertices = np.array(line_vertices, np.int32)
                            rect = cv2.minAreaRect(line_vertices)
                            line_height = min(rect[1])*scale
                            line_width = max(rect[1])*scale
                            font_size = calculate_font_size_for_box(line_str, line_width, line_height, get_font_for_language(args.lang))
                            c.setFont(get_font_for_language(args.lang), max(6, min(12, font_size)))
                            x_coord = max(x_coord, 0)
                            y_coord = max(y_coord, 0)
                            y_coord = min(y_coord, page_height - font_size)
                            c.drawString(x_coord, y_coord, line_str)
                    f.write('\n')
                if c is not None:
                    c.showPage()

                # Increment processed steps and print progress
                processed_steps += 1
                progress = int((processed_steps / total_steps) * 100)
                print(f'PROGRESS:{progress}')
                sys.stdout.flush()

        if c is not None:
            c.save()
    print("Recognition complete.")

    # After all processing is complete and output files are ready, print PROGRESS:100
    print('PROGRESS:100')
    sys.stdout.flush()

    # Clean up temporary directories
    shutil.rmtree(pdf_img_dir)
    shutil.rmtree(combined_json_dir)
    files_to_remove = glob.glob(os.path.join('./test_pdf', '*'))
    for file_rm in files_to_remove:
        os.remove(file_rm)
