package ptithcm.service;

import java.util.List;

import javax.transaction.Transactional;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import ptithcm.entity.KieuSanPhamEntity;
import ptithcm.entity.LoaiSanPhamEntity;
import ptithcm.dao.loaiSanPhamDAO;

@Service
@Transactional
public class loaiSanPhamServiceImpl implements loaiSanPhamService{
	
	@Autowired
	loaiSanPhamDAO loaiDao;
	@Autowired
	KieuSanPhamService kieuService;
	
	@Override
	public LoaiSanPhamEntity layLoaiTheoMa(String maLoai) {
		return loaiDao.layLoaiTheoMa(maLoai);
	}

	@Override
	public List<LoaiSanPhamEntity> layLoai() {
		return loaiDao.layLoai();
	}

	@Override
	public List<LoaiSanPhamEntity> layLoaiTheoGioiTinh(String gioiTinh) {
		return loaiDao.layLoaiTheoGioiTinh(gioiTinh);
	}
	
	@Override
	public boolean kiemTraLoai(String maLoai) {
	    List<KieuSanPhamEntity> danhSachKieu = kieuService.layKieu();
	    
	    for (KieuSanPhamEntity kieu : danhSachKieu) {
	        if (kieu.getLoai().getMaLoai().equals(maLoai)) {
	            return true; // Có kiểu, sản phẩm liên kết
	        }
	    }
	    
	    return false; // Không có kiểu, sản phẩm liên kết
	}

	@Override
	public void themLoai(LoaiSanPhamEntity loai) {
		loaiDao.themLoai(loai);		
	}

	@Override
	public void updateLoai(LoaiSanPhamEntity loai) {
		loaiDao.updateLoai(loai);		
	}

	@Override
	public void xoaLoai(LoaiSanPhamEntity loai) {
		loaiDao.xoaLoai(loai);		
	}



}
