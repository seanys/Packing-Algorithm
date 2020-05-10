from tools.polygon import RatotionPoly,PltFunc,getData
import copy

if __name__ == '__main__':
    polys = getData(6)
    rp = RatotionPoly(90)
    poly_orignal = copy.deepcopy(polys[6])
    poly_rotation = copy.deepcopy(polys[6])
    rp.rotation_specific(poly_rotation)
    PltFunc.addPolygon(poly_orignal)
    PltFunc.addPolygonColor(poly_rotation)
    print(poly_orignal)
    print(poly_rotation)
    PltFunc.showPlt()
    print("success")