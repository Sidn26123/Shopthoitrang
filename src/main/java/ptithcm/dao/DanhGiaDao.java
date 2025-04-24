package ptithcm.dao;

import java.util.List;

import ptithcm.entity.DanhGiaEntity;

public interface DanhGiaDao {
	public List<DanhGiaEntity> layDanhGiaSanPham(String maSp);
	public void saveDanhGia(DanhGiaEntity danhGia);
	public List<DanhGiaEntity> layDanhGiaSanPhamTheoMaND(int maND);
	
}
