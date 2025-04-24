package ptithcm.dao;

import java.util.List;

import ptithcm.entity.HinhAnhEntity;

public interface HinhAnhDAO {
	public void themHinhAnhSanPham(HinhAnhEntity hinhAnh);
	public void suaHinhAnhSanPham(HinhAnhEntity hinhAnh);
	public void xoaHinhAnhSanPham(HinhAnhEntity hinhAnh);
	
}
