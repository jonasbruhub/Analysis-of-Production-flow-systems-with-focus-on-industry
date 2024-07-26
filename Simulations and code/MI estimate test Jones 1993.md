Rho 0.5 - 700x700
2d Trapezoidal w. marginal prob. mass correction
    0.14303874653108195
    0.11008331754888728
    0.12893206254266756
    0.16720284950917774
    0.14268751305958183
    0.12483924590300424
    0.15238775245793748
    0.14067573650190934
    0.19142330529793947
    0.14909811582341922
    0.210385859325786
    0.18750494342624593
    0.20025398066880273
    0.18159424103694724
    0.1736729119309889
    0.15508915139335094
    0.12332034720560818
    0.14611158887897138

2d Simpson  w. marginal prob. mass correction
    0.1430353853177572
    0.11008040729587415
    0.1289249438955533
    0.16719868243366287
    0.14268423142362827
    0.124835075891219
    0.1523828571155624
    0.14066721669126792
    0.19141028881845495
    0.14909162531503511
    0.21037297442423594
    0.18749270281028307
    0.20024748834774883
    0.1815837847855942
    0.1736646380808979
    0.15508709946709862
    0.1233154671114915
    0.1461036523242489



Rho 0.9 - 700x700
2d Trapezoidal w. marginal prob. mass correction
    0.6054481035070927
    0.6479123045150477
    0.6117603202607625
    0.7005614368882138
    0.5748938404581199
    0.6169511762301574
    0.6471515487409124
    0.6186069084596443
    0.6320046485796733
    0.6458366181393687
    0.633018245910793
    0.6277198309747198
    0.6397622796665675  -
    0.6494561993968893
    0.6381136807076354
    0.6354318884074741

2d Simpson  w. marginal prob. mass correction
    0.6054263192994701
    0.6478372227223639
    0.6116042861219462
    0.7004669318829092
    0.574833037330077
    0.6169288324228103
    0.6471162495822916
    0.6185781926713713
    0.6319733409814725
    0.6458083992609298
    0.6329969275411038
    0.626432031494982
    0.6397362112738781  -
    0.649105120150267
    0.6380282946808576
    0.6354105390637385




new method rho 0.9 - was bad for small rho
2d Trapezoidal w. marginal prob. mass correction
    0.7287471279272592
    0.7421588575442155

2d Simpson  w. marginal prob. mass correction
    0.7286326062990287
    0.7420775887181867





# new method rho 0.5 - lambda 0.01
## 2d Trapezoidal w. marginal prob. mass correction
- 0.13683267396018334
- 0.2097030569123254
- 0.16396119436592235
- 0.211819054249364
- 0.1915037527372907
- 0.1804850386607198
- 0.2126198993473928
## 2d Simpson  w. marginal prob. mass correction
- 0.13683024220361267
- 0.2096933355800888
- 0.16395675619156827
- 0.2118112859439282
- 0.1914984501393948
- 0.1804728075386122
- 0.21261594617344742



# new method rho 0.9 - lambda 0.01
## 2d Trapezoidal w. marginal prob. mass correction
- 0.6636340818792599
- 0.5996049180067115
- 0.6474257268615663
- 0.6548441640855592
- 0.6422351376665225
- 0.6469267880608103
## 2d Simpson  w. marginal prob. mass correction
- 0.6636083734911876
- 0.599579506770165
- 0.6474017883013015
- 0.6548248193520201
- 0.6422124517407728
- 0.6469022868498353



# new method rho 0.95 - lambda 0.01
## 2d Trapezoidal w. marginal prob. mass correction
- 0.7500774121743065
## 2d Simpson  w. marginal prob. mass correction
- 0.7500512337925045




# new method rho 0.99 - lambda 0.01
- theoretical MI=1.9585177736258443
## 2d Trapezoidal w. marginal prob. mass correction
- 0.9115695910218243
- 0.9146366138027953
- 0.9158568004969743
- 0.9223080320551285
- 0.9304208436383549

## 2d Simpson  w. marginal prob. mass correction
- 0.9115384662871291
- 0.9146051740109312
- 0.9158269610002103
- 0.9222769671185495
- 0.9303897770714079




# new method rho 0.995 - lambda 0.01
- theoretical MI=2.303836658103107
## 2d Trapezoidal w. marginal prob. mass correction
- 0.9403536839895259
- 0.9469261025876767
- 0.943671300696212
- 0.9392728291057852
- 0.9373989770760539

## 2d Simpson  w. marginal prob. mass correction
- 0.9403215772238922
- 0.9468948632084443
- 0.9436394127970125
- 0.9392406607771709
- 0.9373682157638153







<!-- Is very stable for large correlations even. But for large correlations, should be more of a strait line, but is too wide which could be from badly estimated (0,0) and (1,1) which when rescaling makes it seem too wide. By having a smaller bandwith here, the probability mass would be less but high MI and hence when rescaling would have a more similar to the theoretical. The too wide (0,0) and (1,1) can also be seen in marginals where they are not uniform.

Refer to regEM from diffusion or one could use K-means clusetering to est. the local variance etc.

For example, for rho = 0.995, multiplying h by 0.3 gets much better results than standard h from scott, but for rho = 0.5, this leads to dramatic overestimation : 0.29900026438141497, actually 0.14384103622589045
-->

<!-- for more dense areas should have smaller bandwidth -> optimally, we would have a local bandwidth -->