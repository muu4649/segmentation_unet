import cv2  #OpenCVのインポート

fname="predict_data.png" #開く画像ファイル名
threshold=30 #二値化閾値

img_color= cv2.imread(fname) #画像を読み出しオブジェクトimg_colorに代入
img_gray = cv2.imread(fname,cv2.IMREAD_GRAYSCALE) #画像をグレースケールで読み出しオブジェクトimg_grayに代入

ret, img_binary= cv2.threshold(img_gray, threshold, 255, cv2.THRESH_BINARY) #オブジェクトimg_grayを閾値threshold(127)で二値化しimg_binaryに代入
contours, hierarchy = cv2.findContours(img_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #img_binaryを輪郭抽出
cv2.drawContours(img_color, contours, -1, (0,0,255), 2) #抽出した輪郭を赤色でimg_colorに重ね書き
print(len(contours)) #抽出した輪郭の個数を表示する

#cv2.imshow("contours",img_color) #別ウィンドウを開き(ウィンドウ名 "contours")オブジェクトimg_colorを表示
#cv2.waitKey(0) #キー入力待ち
#cv2.destroyAllWindows() #ウインドウを閉じる
