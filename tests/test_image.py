import pymupdf as fitz
from pathlib import Path


pdf_path = "uploads/unillm_image_extraction_test.pdf"
output_dir = Path("extracted_images")

output_dir.mkdir(exist_ok=True)


pdf = fitz.open(pdf_path)

image_count = 0


for page_number, page in enumerate(pdf, start=1):

    images = page.get_images(full=True)

    print(
        f"Page {page_number}: "
        f"{len(images)} embedded image(s)"
    )

    for image_index, image in enumerate(images, start=1):

        xref = image[0]

        image_info = pdf.extract_image(xref)

        image_bytes = image_info["image"]
        image_ext = image_info["ext"]

        image_count += 1

        output_path = (
            output_dir
            / f"page_{page_number}_image_{image_index}.{image_ext}"
        )

        with open(output_path, "wb") as image_file:
            image_file.write(image_bytes)

        print(
            f"  Extracted: {output_path}"
        )


pdf.close()


print()
print(
    f"Total images extracted: {image_count}"
)