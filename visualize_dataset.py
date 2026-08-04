import cv2
from src.data import load_datasets

train_data, _ = load_datasets()

index = 0

while True:
    row = train_data.iloc[index]

    image = cv2.imread(str(row["image_path"]))

    image = cv2.resize(image, (960, 540))

    cv2.putText(
        image,
        f"Index: {index}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0),
        2,
    )

    cv2.putText(
        image,
        f"Steering: {row['Steering']:.3f}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0),
        2,
    )

    cv2.imshow("Dataset", image)

    key = cv2.waitKey(0)

    if key == ord("d"):
        index += 1

    elif key == ord("a"):
        index -= 1

    elif key == 27:
        break

cv2.destroyAllWindows()