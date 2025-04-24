package ptithcm.dao;

import java.util.List;

import ptithcm.entity.KieuSanPhamEntity;

public interface KieuSanPhamDAO {
	public List<KieuSanPhamEntity> layKieu();
	public List<KieuSanPhamEntity> layKieuTheoLoai(String loai);
	public KieuSanPhamEntity layKieuTheoMa(String maKieu);
	public KieuSanPhamEntity layKieuTheoMa(int maKieu);
	
	public void themKieu(KieuSanPhamEntity Kieu);
	public void updateKieu(KieuSanPhamEntity Kieu);
	public void xoaKieu(KieuSanPhamEntity Kieu);
	
}
