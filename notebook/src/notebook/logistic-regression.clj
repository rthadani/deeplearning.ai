;; gorilla-repl.fileformat = 1

;; **
;;; ©ß# Gorilla REPL
;;; 
;;; Welcome to gorilla :-)
;;; 
;;; Shift + enter evaluates code. Hit ctrl+g twice in quick succession or click the menu icon (upper-right corner) for more commands ...
;;; 
;;; It's a good habit to run each worksheet in its own namespace: feel free to use the declaration we've provided below if you'd like.
;; **

;; @@
(ns notebook.logistic-regression
  (:require [gorilla-plot.core :as plot]
            [mikera.image.core :as i]
            [clojure.java.io :refer [resource]])
  (:use [uncomplicate.neanderthal core native vect-math]
        [notebook numpy]))
;; @@
;; =>
;;; {"type":"html","content":"<span class='clj-nil'>nil</span>","value":"nil"}
;; <=

;; **
;;; Forward and Backward propagation steps for learning the parameters.
;;; 
;;; 
;;; Forward Propagation:
;;; - You get X
;;; - You compute 
;;; $$ A = \sigma(w^T X + b) = (a^{(0)}, a^{(1)}, ..., a^{(m-1)}, a^{(m)}) $$
;;; - You calculate the cost function: 
;;; $$ J = -\frac{1}{m}\sum_{i=1}^{m}y^{(i)}\log(a^{(i)})+(1-y^{(i)})\log(1-a^{(i)})$$
;;; 
;;; Here are the two formulas: 
;;; 
;;; $$\frac{\partial J}{\partial w} = \frac{1}{m}X(A-Y)^T\tag{7}$$
;;; $$\frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^m (a^{(i)}-y^{(i)})\tag{8}$$
;; **

;; @@
(defn sigmoid-activation
  [Z]
  (let [one+e-x (bcast-plus (exp (scal -1 Z)) 1)]
    (div! (broadcast one+e-x 1) one+e-x)))

(defn cost [A Y m]
  (let [ylogyhat (mul Y (log A))
        one-y*logone-yhat (-> (bcast-plus (scal -1 Y) 1)
                         (bcast-mul (log (bcast-plus (scal -1 A) 1))))]
    (/ (* -1 (sum(bcast-plus ylogyhat one-y*logone-yhat))) m))) 

(defn gradients
  [nxm W b Y]
  (let [m    (ncols nxm)
        wt   (trans W)
        Z    (bcast-plus (mm wt nxm) b) 
        A    (sigmoid-activation Z)
        dz   (axpy -1 Y A)
        dw   (scal (/ 1 m) (mm nxm (trans dz)))
        db   (/ (sum dz) m)]
      
       [dw db (cost A Y m)]))

(defn logistic-regression
  [nxm W b Y alpha iterations] 
  (loop [W       W
         b       b 
        [dw db cost] (gradients nxm W b Y)
        iter 0]
    (if (= iter iterations)
      [W b cost]
      (let [W (axpy (* -1 alpha) dw W) 
            b  (- b (* alpha db))
           [dw db cost] (gradients nxm W b Y)]
        (recur W b [dw db cost] (inc iter))))))

(defn predict
  [W b X]
  (let [m (ncols X)
        Z (bcast-plus (mm (trans W) X) b)
        A (sigmoid-activation Z)]
    (alter! A (fn ^double [^long _ ^long _ ^double a] (if (> a 0.5) 1.0 0.0)))))
;; @@
;; =>
;;; {"type":"html","content":"<span class='clj-var'>#&#x27;notebook.logistic-regression/predict</span>","value":"#'notebook.logistic-regression/predict"}
;; <=

;; @@
(def w (dge 2 1 [1 2])) 
(def b 2)
(def X (dge 2 2 [1 2 3 4] {:layout :row}))
(def Y (dge 1 2 [1 0] {:layout :row}))

;(activation X (trans w) b)
(let [[wr br cost] (logistic-regression X w b Y 0.009 100)]
  [wr br cost (predict wr br X)])

;; @@
;; =>
;;; {"type":"list-like","open":"<span class='clj-vector'>[</span>","close":"<span class='clj-vector'>]</span>","separator":" ","items":[{"type":"html","content":"<span class='clj-unkown'>#RealGEMatrix[double, mxn:2x1, layout:column, offset:0]\n   ▥       ↓       ┓    \r\n   →       0.11         \n   →       0.23         \n   ┗               ┛    \n</span>","value":"#RealGEMatrix[double, mxn:2x1, layout:column, offset:0]\n   ▥       ↓       ┓    \r\n   →       0.11         \n   →       0.23         \n   ┗               ┛    \n"},{"type":"html","content":"<span class='clj-double'>1.5593049248448891</span>","value":"1.5593049248448891"},{"type":"html","content":"<span class='clj-double'>1.4313999565615696</span>","value":"1.4313999565615696"},{"type":"html","content":"<span class='clj-unkown'>#RealGEMatrix[double, mxn:1x2, layout:row, offset:0]\n   ▤       ↓       ↓       ┓    \r\n   →       1.00    1.00         \n   ┗                       ┛    \n</span>","value":"#RealGEMatrix[double, mxn:1x2, layout:row, offset:0]\n   ▤       ↓       ↓       ┓    \r\n   →       1.00    1.00         \n   ┗                       ┛    \n"}],"value":"[#RealGEMatrix[double, mxn:2x1, layout:column, offset:0]\n   ▥       ↓       ┓    \r\n   →       0.11         \n   →       0.23         \n   ┗               ┛    \n 1.5593049248448891 1.4313999565615696 #RealGEMatrix[double, mxn:1x2, layout:row, offset:0]\n   ▤       ↓       ↓       ┓    \r\n   →       1.00    1.00         \n   ┗                       ┛    \n]"}
;; <=

;; @@
(def img (-> "images/2layerNN_kiank.png" resource i/load-image))
;; @@
;; =>
;;; {"type":"html","content":"<span class='clj-var'>#&#x27;notebook.logistic-regression/img</span>","value":"#'notebook.logistic-regression/img"}
;; <=

;; @@

;; @@
;; =>
;;; {"type":"html","content":"<span class='clj-unkown'>-5406930</span>","value":"-5406930"}
;; <=

;; @@

;; @@
